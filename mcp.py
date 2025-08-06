# mcp.py

import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from meteomatics import api
import openrouteservice
from openrouteservice import exceptions
import pandas as pd
import httpx
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI


load_dotenv()

"""
# app = FastAPI(title="Smart Weather MCP متكامل")

# إعدادات CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "file://",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""

app = FastAPI(title="MCP Weather and so on")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to MCP Weather API"} 


# تهيئة Geocoder من Nominatim لتحديد الموقع العكسي للنقاط المأخوذة
geolocator_reverse = Nominatim(user_agent="smart-weather-mcp-app-reverse")

# URL الخاص بـ NCM EWS API
NCM_EWS_API_URL = "https://meteo.ncm.gov.sa/public/ews/latest.json"

# ---------- نماذج الطلب ----------
class TravelRequest(BaseModel):
    start_coords: List[float]  # [خط عرض, خط طول]
    end_coords: List[float]    # [خط عرض, خط طول]
    departure_time: str        # ISO8601
    start_city: str            # اسم مدينة البداية (للملخص)
    end_city: str              # اسم مدينة النهاية (للملخص)

# ---------- MCP descriptor ----------
@app.get("/.well-known/mcp")
def mcp_descriptor():
    return {
        "name": "Smart Weather MCP Integrated",
        "version": "1.0",
        "services": {
            "plan_travel_weather_route": {
                "description": "مسار بين نقطتين مع توقعات الطقس وتحذيرات موحدة من Meteomatics والهيئة الوطنية.",
                "inputs": ["start_coords", "end_coords", "departure_time", "start_city", "end_city"],
                "outputs": ["route_summary", "route_map", "weather_along_route_with_alerts"]
            }
        }
    }

# ---------- دوال التخزين المؤقت ----------
@lru_cache(maxsize=128)
def get_national_alerts_cached():
    return get_national_alerts_raw()

# ---------- ترجمة ذكية بسيطة ----------
EN_TO_AR_SIMPLE = {
    "rain": "مطر", "storm": "عاصفة", "warning": "تحذير", "high": "عالي",
    "moderate": "متوسط", "low": "منخفض", "strong": "قوي", "wind": "رياح",
    "fog": "ضباب", "الإنذار الأحمر": "عالي", "الإنذار البرتقالي": "متوسط",
    "الإنذار الأصفر": "منخفض", "الإنذار الأخضر": "منخفض",
    "Watch": "متوسط", "Warning": "عالي", "New Alert": "غير معروف"
}

def is_arabic(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))

def smart_translate(text: str, target_lang="ar") -> str:
    if target_lang == "ar":
        if is_arabic(text):
            for ar_phrase, unified_level in EN_TO_AR_SIMPLE.items():
                if ar_phrase in text:
                    return unified_level
            return text
        lowered = text.lower()
        for en, ar in EN_TO_AR_SIMPLE.items():
            lowered = re.sub(r'\b' + re.escape(en) + r'\b', ar, lowered)
        return lowered.capitalize()
    else:
        return text

# ---------- مصادر البيانات ----------

def query_time_series_in_chunks(
    coords: List[tuple],
    chunk_size: int,
    start_date: datetime,
    end_date: datetime,
    interval: timedelta,
    parameters: List[str],
    username: str,
    password: str
) -> pd.DataFrame:
    """
    Queries Meteomatics API in chunks to avoid URL length limits.
    Returns a single pandas DataFrame with MultiIndex (lat, lon, time).
    """
    all_dfs = []
    for i in range(0, len(coords), chunk_size):
        chunk = coords[i:i + chunk_size]
        try:
            df_chunk = api.query_time_series(chunk, start_date, end_date, interval, parameters, username, password)
            all_dfs.append(df_chunk)
        except Exception as e:
            print(f"Error fetching Meteomatics data for chunk {i}-{i+chunk_size}: {e}")
            continue
    if all_dfs:
        return pd.concat(all_dfs)
    return pd.DataFrame()


def get_national_alerts_raw() -> List[Dict[str, Any]]:
    url = NCM_EWS_API_URL
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"فشل جلب الإنذارات الوطنية من NCM API: {e}. قد يكون API المركز الوطني غير متاح أو هناك مشكلة في الاتصال.")
    except Exception as e:
        raise RuntimeError(f"حدث خطأ غير متوقع أثناء جلب الإنذارات الوطنية: {e}")

def extract_alerts_for_location(national_data: List[Dict[str, Any]], location_name: str) -> List[Dict[str, Any]]:
    """
    تستخرج الإنذارات ذات الصلة من بيانات NCM بناءً على اسم الموقع.
    تفترض أن national_data هي قائمة من قواميس الإنذارات من NCM EWS API.
    """
    alerts = []
    lower_location_name = location_name.lower()

    for alert_item in national_data:
        region_ar = alert_item.get("regionAR", "").lower()
        governorates = alert_item.get("governorates", [])
        
        if lower_location_name in region_ar or region_ar in lower_location_name:
            hazard_messages = [h.get("descriptionAr", "") for h in alert_item.get("alertHazards", []) if h.get("descriptionAr")]
            full_message = f"{alert_item.get('alertStatusAr', '')}"
            if hazard_messages:
                full_message += f" ({', '.join(hazard_messages)})"

            alerts.append({
                "level": smart_translate(alert_item.get("alertTypeAr", "غير معروف")),
                "message": full_message,
                "recommendation": "",
                "source": "المركز الوطني للأرصاد"
            })
            continue

        for gov in governorates:
            gov_name_ar = gov.get("nameAr", "").lower()
            if lower_location_name in gov_name_ar or gov_name_ar in lower_location_name:
                hazard_messages = [h.get("descriptionAr", "") for h in alert_item.get("alertHazards", []) if h.get("descriptionAr")]
                full_message = f"{alert_item.get('alertStatusAr', '')}"
                if hazard_messages:
                    full_message += f" ({', '.join(hazard_messages)})"

                alerts.append({
                    "level": smart_translate(alert_item.get("alertTypeAr", "غير معروف")),
                    "message": full_message,
                    "recommendation": "",
                    "source": "المركز الوطني للأرصاد"
                })
                break
    return alerts


# ---------- تنبيهات محلية مبنية من بيانات الطقس ----------
def get_local_inferred_alerts(forecast_df: pd.DataFrame) -> List[Dict[str, Any]]:
    alerts = []
    try:
        if not forecast_df.empty:
            precip = forecast_df['Precipitation (mm)'].sum()
            wind_speed_avg = forecast_df['Wind Speed (m/s)'].mean()
            temp_avg = forecast_df['Temperature (C)'].mean()
            if precip >= 10:
                alerts.append({
                    "level": "عالي",
                    "message": "أمطار غزيرة متوقعة.", 
                    "recommendation": "تجنّب القيادة بسرعة وابقَ تحت مغطى.",
                    "source": "مشتق محلي"
                })
            elif precip >= 2:
                alerts.append({
                    "level": "متوسط",
                    "message": "أمطار متوسطة.", 
                    "recommendation": "احمل مظلة.",
                    "source": "مشتق محلي"
                })
            if wind_speed_avg >= 15:
                alerts.append({
                    "level": "متوسط",
                    "message": "رياح قوية.", 
                    "recommendation": "ثبت الأجسام الخفيفة.",
                    "source": "مشتق محلي"
                })
            if temp_avg >= 45:
                alerts.append({
                    "level": "عالي",
                    "message": "حرارة مرتفعة جداً.", 
                    "recommendation": "تجنّب التعرض المباشر للشمس.",
                    "source": "مشتق محلي"
                })
    except Exception as e:
        print(f"Error in get_local_inferred_alerts: {e}")
        pass
    return alerts

# ---------- دمج و تقييم مستوى الخطر موحد ----------
LEVEL_SCORE = {
    "منخفض": 1, "متوسط": 2, "عالي": 3, "high": 3, "moderate": 2, "low": 1, "غير معروف": 1
}

def normalize_level(level: str) -> str:
    l = level.strip().lower()
    if l in ("عالي", "high", "الإنذار الأحمر", "warning"):
        return "عالي"
    if l in ("متوسط", "moderate", "الإنذار البرتقالي", "watch"):
        return "متوسط"
    if l in ("منخفض", "low", "الإنذار الأصفر", "الإنذار الأخضر"):
        return "منخفض"
    return "غير معروف"

def unified_risk_level(all_alerts: List[Dict[str, Any]]) -> str:
    highest = 0
    for a in all_alerts:
        lvl = normalize_level(a.get("level", "غير معروف"))
        score = LEVEL_SCORE.get(lvl, 1)
        if score > highest:
            highest = score
    for k, v in LEVEL_SCORE.items():
        if v == highest:
            return k if k in ("عالي", "متوسط", "منخفض") else "غير معروف"
    return "غير معروف"

# ---------- تخطيط الرحلة مع الطقس والإنذارات ----------
@app.post("/plan_travel_weather_route")
def get_route_and_weather(req: TravelRequest) -> Dict[str, Any]:
    start_coords = req.start_coords
    end_coords = req.end_coords
    departure_iso = req.departure_time
    start_city = req.start_city
    end_city = req.end_city

    # 1. المسار
    ors_key = os.getenv("OPENROUTESERVICE_API_KEY")
    if not ors_key:
        raise HTTPException(status_code=500, detail="مفتاح OpenRouteService غير موجود في .env. يرجى التحقق من ملف .env.")
    client = openrouteservice.Client(key=ors_key)
    route = None
    try:
        route = client.directions(
            coordinates=[(start_coords[1], start_coords[0]), (end_coords[1], end_coords[0])],
            profile='driving-car',
            format='geojson'
        )
    except exceptions.ApiError as e:
        error_detail = f"فشل تخطيط المسار: {e}"
        if hasattr(e, 'args') and len(e.args) > 0 and isinstance(e.args[0], dict):
            ors_error = e.args[0].get('error', {})
            error_detail = ors_error.get('message', error_detail)
        raise HTTPException(status_code=400, detail=error_detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ غير متوقع أثناء الاتصال بخدمة تخطيط المسار: {e}")

    if not route or 'features' not in route or not route['features']:
        raise HTTPException(status_code=400, detail="لم يتم العثور على مسار صالح بين المدينتين المحددتين. يرجى التحقق من أسماء المدن أو وجود طريق قابل للتوجيه بينهما.")

    coords = route['features'][0]['geometry']['coordinates']
    # أخذ عينات من النقاط على طول المسار لتقليل عدد طلبات الطقس
    sampled_route_coords = [(pt[1], pt[0]) for pt in coords][::5]

    # Meteomatics parameters and date range
    parameters = [
        "t_2m:C",
        "precip_1h:mm",
        "wind_dir_10m:d",
        "wind_speed_10m:ms"
    ]
    start_date_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end_date_utc = start_date_utc + timedelta(days=10)
    interval = timedelta(hours=1)

    username = os.getenv("METEOMATICS_API_USERNAME")
    password = os.getenv("METEOMATICS_API_PASSWORD")
    if not username or not password:
        raise HTTPException(status_code=500, detail="مفاتيح Meteomatics غير معرفة في .env. يرجى التحقق من ملف .env.")

    full_forecast_df = pd.DataFrame()
    try:
        full_forecast_df = query_time_series_in_chunks(
            sampled_route_coords,
            50,
            start_date_utc,
            end_date_utc,
            interval,
            parameters,
            username,
            password
        )
        
        # إعادة تسمية الأعمدة بعد الجلب. هذه الأعمدة هي بيانات وليست جزءًا من الفهرس.
        full_forecast_df = full_forecast_df.rename(columns={
            't_2m:C': 'Temperature (C)',
            'precip_1h:mm': 'Precipitation (mm)',
            'wind_dir_10m:d': 'Wind Direction (degrees)',
            'wind_speed_10m:ms': 'Wind Speed (m/s)'
        })
        
        # تأكد من أن الفهرس لا يزال multi-index (lat, lon, time)
        # إذا كانت الدالة query_time_series_in_chunks تُرجع بالفعل DataFrame بفهرس متعدد المستويات،
        # فلا حاجة لـ reset_index ثم set_index هنا.
        # فقط تأكد من أن الأعمدة التي تم إعادة تسميتها موجودة.
        
        print("Full forecast DataFrame head:")
        print(full_forecast_df.head())

    except RuntimeError as e:
        print(f"Error fetching full forecast: {e}")
        raise HTTPException(status_code=500, detail=f"فشل جلب توقعات الطقس للمسار: {e}")


    national_data = get_national_alerts_cached()

    weather_along = {}
    for i, (lat, lon) in enumerate(sampled_route_coords):
        point_key = f"point_{i}"
        entry: Dict[str, Any] = {"coord": [lat, lon]}
        
        point_location_name = ""
        try:
            reverse_location = geolocator_reverse.reverse(f"{lat}, {lon}", timeout=5, language='ar')
            if reverse_location and reverse_location.address:
                address_parts = reverse_location.raw.get('address', {})
                point_location_name = address_parts.get('city') or address_parts.get('town') or \
                                      address_parts.get('village') or address_parts.get('county') or \
                                      address_parts.get('state') or ""
                print(f"Reverse geocoded point {i} to: {point_location_name}")
        except Exception as e:
            print(f"خطأ في تحديد الموقع العكسي للنقطة {lat},{lon}: {e}")
            point_location_name = ""

        try:
            # استخراج بيانات الطقس للنقطة الحالية من DataFrame الكامل
            # نستخدم .loc مع الفهرس المتعدد المستويات (lat, lon)
            # ثم نختار أول شريحة زمنية (iloc[[0]])
            point_forecast_df = pd.DataFrame() # تهيئة DataFrame فارغ
            if (lat, lon) in full_forecast_df.index:
                # Get all time slices for this lat/lon, then take the first one
                point_forecast_df = full_forecast_df.loc[(lat, lon)].iloc[[0]]
            else:
                print(f"No forecast data found for point: {lat}, {lon}")

            summary = summarize_forecast(point_forecast_df, language="ar")
            entry["forecast_summary"] = summary["summary"]
            entry["forecast_card"] = summary["card"]
            
            national_alerts = extract_alerts_for_location(national_data, point_location_name)
            local_alerts = get_local_inferred_alerts(point_forecast_df)
            all_alerts = national_alerts + local_alerts
            entry["alerts"] = all_alerts
            entry["unified_risk"] = unified_risk_level(all_alerts)
        except RuntimeError as e:
            entry["error"] = str(e)
            print(f"Error processing point {point_key}: {e}")
        except Exception as e:
            entry["error"] = str(e)
            print(f"An unexpected error occurred for point {point_key}: {e}")
        weather_along[point_key] = entry

    route_summary = f"رحلة من {start_city} إلى {end_city} بتاريخ {departure_iso}، مع توقعات الطقس وإنذارات موحدة."
    return {
        "route_summary": route_summary,
        "route_map": route,
        "weather_along_route": weather_along
    }

# ---------- دالة التلخيص المساعدة ----------
def summarize_forecast(df: pd.DataFrame, language="ar"):
    if df.empty:
        return {"summary": "لا توجد بيانات طقس متاحة.", "card": {
            "temperature": "-", "precipitation": "-", "wind_speed": "-", "icon": "❓"
        }}

    temps = df['Temperature (C)']
    precip = df['Precipitation (mm)']
    wind = df['Wind Speed (m/s)']
    avg_temp = temps.mean()
    total_precip = precip.sum()
    avg_wind = wind.mean()
    if language == "ar":
        summary = (
            f"متوسط الحرارة {avg_temp:.1f}°م، "
            f"إجمالي الأمطار {total_precip:.1f} ملم، "
            f"سرعة الرياح {avg_wind:.1f} م/ث."
        )
    else:
        summary = (
            f"Avg temp {avg_temp:.1f}°C, "
            f"Total precipitation {total_precip:.1f} mm, "
            f"Wind speed {avg_wind:.1f} m/s."
        )
    card = {
        "temperature": f"{avg_temp:.1f}°C",
        "precipitation": f"{total_precip:.1f} mm",
        "wind_speed": f"{avg_wind:.1f} m/s",
        "icon": "🌦"
    }
    return {"summary": summary, "card": card}

# ---------- تشغيل ---------
# من سطر الأوامر: uvicorn mcp:app --reload
