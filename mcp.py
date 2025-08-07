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

from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI(title="Smart Weather MCP متكامل")

@app.get("/")
def read_root():
    return {"Hello": "World"}
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "file://",
    "null"
    "https://ayahbfallatah.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- نماذج الطلب ----------
class TravelRequest(BaseModel):
    start_city: str
    end_city: str
    departure_time: str

# ---------- MCP descriptor ----------
@app.get("/.well-known/mcp")
def mcp_descriptor():
    return {
        "name": "Smart Weather MCP Integrated",
        "version": "1.0",
        "services": {
            "plan_travel_weather_route": {
                "description": "مسار بين نقطتين مع توقعات الطقس وتحذيرات موحدة من Meteomatics والهيئة الوطنية.",
                "inputs": ["start_city", "end_city", "departure_time"],
                "outputs": ["route_summary", "route_map", "weather_along_route"]
            }
        }
    }

# ---------- دوال التخزين المؤقت ----------
@lru_cache(maxsize=128)
def get_national_alerts_cached():
    return get_national_alerts_raw()

# ---------- ترجمة ذكية بسيطة ----------
EN_TO_AR_SIMPLE = {
    "rain": "مطر",
    "storm": "عاصفة",
    "warning": "تحذير",
    "high": "عالي",
    "moderate": "متوسط",
    "low": "منخفض",
    "strong": "قوي",
    "wind": "رياح",
    "fog": "ضباب",
    "الإنذار الأحمر": "عالي",
    "الإنذار البرتقالي": "متوسط",
    "الإنذار الأصفر": "منخفض",
    "الإنذار الأخضر": "منخفض",
    "Watch": "متوسط",
    "Warning": "عالي",
    "New Alert": "غير معروف"
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

def get_forecast_meteomatics(coords: List[List[float]], start_iso: str, end_iso: str, interval_hours: int = 1) -> pd.DataFrame:
    username = os.getenv("METEOMATICS_API_USERNAME")
    password = os.getenv("METEOMATICS_API_PASSWORD")
    if not username or not password:
        raise RuntimeError("مفاتيح Meteomatics غير معرفة في .env. يرجى التحقق من ملف .env.")
    
    start_date = datetime.fromisoformat(start_iso)
    end_date = datetime.fromisoformat(end_iso)
    interval = timedelta(hours=interval_hours)
    
    parameters = [
        "t_2m:C",
        "precip_1h:mm",
        "wind_dir_10m:d",
        "wind_speed_10m:ms"
    ]
    
    try:
        df = api.query_time_series(coords, start_date, end_date, interval, parameters, username, password)
        df = df.reset_index()
        df = df.rename(columns={
            'validdate': 'time',
            'lat': 'latitude',
            'lon': 'longitude',
            't_2m:C': 'Temperature (C)',
            'precip_1h:mm': 'Precipitation (mm)',
            'wind_dir_10m:d': 'Wind Direction (deg)',
            'wind_speed_10m:ms': 'Wind Speed (m/s)'
        })
        df = df.set_index(['latitude', 'longitude', 'time'])
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل جلب بيانات الطقس من Meteomatics: {e}")

def get_national_alerts_raw() -> List[Dict[str, Any]]:
    url = "https://meteo.ncm.gov.sa/public/ews/latest.json"
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise RuntimeError(f"فشل جلب الإنذارات الوطنية: {e}. قد يكون API المركز الوطني غير متاح أو هناك مشكلة في الاتصال.")

def extract_alerts_for_location(national_data: List[Dict[str, Any]], location_name: str) -> List[Dict[str, Any]]:
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


def get_local_inferred_alerts(forecast_df: pd.DataFrame) -> List[Dict[str, Any]]:
    alerts = []
    try:
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
    except Exception:
        pass
    return alerts


LEVEL_SCORE = {
    "منخفض": 1,
    "متوسط": 2,
    "عالي": 3,
    "high": 3,
    "moderate": 2,
    "low": 1,
    "غير معروف": 1
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

# Function to get coordinates from city name
def get_coordinates(city_name: str) -> List[float]:
    geolocator = Nominatim(user_agent="smart-weather-mcp-app-geocoder")
    try:
        location = geolocator.geocode(city_name, timeout=5)
        if location:
            return [location.latitude, location.longitude]
        raise HTTPException(status_code=400, detail=f"لم يتم العثور على إحداثيات للمدينة: {city_name}")
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        raise HTTPException(status_code=500, detail=f"خطأ في خدمة تحديد المواقع: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ غير متوقع أثناء تحديد الموقع: {e}")

# Hardcoded list of major cities
MAJOR_CITIES = {
    "جدة": [21.4858, 39.1925],
    "الرياض": [24.7136, 46.6753],
    "الدمام": [26.4207, 50.0888],
    "مكة المكرمة": [21.3891, 39.8579],
    "المدينة المنورة": [24.5247, 39.5692],
    "الطائف": [21.2709, 40.4144],
    "أبها": [18.2166, 42.5033],
    "تبوك": [28.3995, 36.5746],
    "بريدة": [26.3260, 43.9749]
}

def summarize_forecast(df: pd.DataFrame, language="ar"):
    try:
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
    except Exception:
        summary = "لا توجد بيانات متاحة."
        card = {
            "temperature": "-",
            "precipitation": "-",
            "wind_speed": "-",
            "icon": "❓"
        }
    return {"summary": summary, "card": card}

@app.post("/plan_travel_weather_route")
def get_route_and_weather(req: TravelRequest) -> Dict[str, Any]:
    # Use start_city from the request instead of hardcoding
    start_city = req.start_city
    end_city = req.end_city
    departure_iso = req.departure_time

    start_coords = MAJOR_CITIES.get(start_city)
    end_coords = MAJOR_CITIES.get(end_city)
    
    if not start_coords or not end_coords:
        raise HTTPException(status_code=400, detail="الرجاء اختيار مدينة بداية ونهاية صالحة من القائمة.")

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
    total_duration_sec = route['features'][0]['properties']['summary']['duration']
    
    num_points = 10
    sampled_coords = [(pt[1], pt[0]) for pt in coords][::max(1, len(coords) // num_points)]
    
    segment_length = total_duration_sec / len(sampled_coords) if len(sampled_coords) > 0 else 0

    national_data = get_national_alerts_cached()
    all_alerts_list = []
    weather_along = []

    geolocator_reverse = Nominatim(user_agent="smart-weather-mcp-app-reverse")

    for i, (lat, lon) in enumerate(sampled_coords):
        arrival_time_offset = timedelta(seconds=i * segment_length)
        arrival_time = datetime.fromisoformat(departure_iso) + arrival_time_offset
        
        entry: Dict[str, Any] = {
            "coord": [lat, lon],
            "arrival_time": arrival_time.isoformat()
        }
        
        try:
            forecast_df = get_forecast_meteomatics(
                [[lat, lon]], 
                (arrival_time - timedelta(minutes=30)).isoformat(), 
                (arrival_time + timedelta(minutes=30)).isoformat(), 
                1
            )
            
            df_point = forecast_df.loc[(lat, lon)]
            summary = summarize_forecast(df_point, language="ar")
            entry["forecast_summary"] = summary["summary"]
            entry["forecast_card"] = summary["card"]
            
            point_location_name = ""
            try:
                reverse_location = geolocator_reverse.reverse(f"{lat}, {lon}", timeout=5, language='ar')
                if reverse_location and reverse_location.address:
                    address_parts = reverse_location.raw.get('address', {})
                    point_location_name = address_parts.get('city') or address_parts.get('town') or \
                                          address_parts.get('village') or address_parts.get('county') or \
                                          address_parts.get('state') or ""
            except Exception:
                point_location_name = ""
            
            entry["location_name"] = point_location_name

            national_alerts = extract_alerts_for_location(national_data, point_location_name)
            local_alerts = get_local_inferred_alerts(df_point)
            
            all_point_alerts = national_alerts + local_alerts
            
            for alert in all_point_alerts:
                alert['location_name'] = point_location_name
                all_alerts_list.append(alert)

            entry["alerts"] = all_point_alerts
            entry["unified_risk"] = unified_risk_level(all_point_alerts)
            
        except Exception as e:
            entry["error"] = str(e)
            entry["forecast_summary"] = "فشل في جلب بيانات الطقس لهذه النقطة."
            entry["alerts"] = []
            entry["unified_risk"] = "غير معروف"
            entry["location_name"] = ""

        weather_along.append(entry)

    overall_unified_risk = unified_risk_level(all_alerts_list)
    
    route_summary = f"توقعات الطقس التفصيلية للمسار من {start_city} إلى {end_city}، تبدأ في {departure_iso}."
    
    return {
        "route_summary": route_summary,
        "overall_risk": overall_unified_risk,
        "route_map": route,
        "weather_along_route": weather_along,
        "alerts_log": all_alerts_list
    }
# ---------- تشغيل ---------
# من سطر الأوامر: uvicorn mcp:app --reload
# لتشغيل التطبيق
