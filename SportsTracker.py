import pandas as pd
import numpy as np
import streamlit as st
from scipy.fft import fft
from scipy.signal import butter, filtfilt, find_peaks
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

location_data = load_data('location.csv')
acceleration_data = load_data('acceleration.csv')

def low_pass_filter(data, cutoff=1.5, fs=50, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

@st.cache_data
def cached_compute_filtered_steps(acceleration, threshold_factor=2.0, min_peak_distance=30):
    filtered_acceleration = low_pass_filter(acceleration, cutoff=1.5)
    threshold = np.mean(filtered_acceleration) + threshold_factor * np.std(filtered_acceleration)
    peaks, _ = find_peaks(filtered_acceleration, height=threshold, distance=min_peak_distance)
    return len(peaks)

@st.cache_data
def cached_compute_fourier_steps(acceleration, sampling_rate=50):
    power_spectrum = np.abs(fft(acceleration))**2
    freqs = np.fft.fftfreq(len(acceleration), d=1/sampling_rate)
    positive_freqs = freqs > 0
    step_frequency = freqs[positive_freqs][np.argmax(power_spectrum[positive_freqs])]
    estimated_steps = step_frequency * len(acceleration) / sampling_rate
    return estimated_steps

def compute_distance_speed(location_data):
    coords = list(zip(location_data['Latitude (°)'], location_data['Longitude (°)']))
    distances = [geodesic(coords[i], coords[i+1]).meters for i in range(len(coords)-1)]
    total_distance = sum(distances) / 1000
    total_time = location_data['Time (s)'].iloc[-1] - location_data['Time (s)'].iloc[0]
    average_speed_m_s = total_distance * 1000 / total_time if total_time > 0 else 0
    average_speed_km_h = average_speed_m_s * 3.6
    return total_distance, average_speed_m_s, average_speed_km_h

st.title("Päivän liikunta")

filtered_steps = cached_compute_filtered_steps(acceleration_data['Linear Acceleration y (m/s^2)'], threshold_factor=1.5)
fourier_steps = cached_compute_fourier_steps(acceleration_data['Linear Acceleration y (m/s^2)'])
total_distance, average_speed_m_s, average_speed_km_h = compute_distance_speed(location_data)
average_step_length = (total_distance * 1000) / fourier_steps * 100

st.write(f"Askelmäärä laskettuna suodatuksen avulla: {filtered_steps:.0f} askelta")
st.write(f"Askelmäärä laskettuna Fourier-analyysin avulla: {fourier_steps:.0f} askelta")
st.write(f"Keskinopeus: {average_speed_m_s:.2f} m/s (tai {average_speed_km_h:.2f} km/h)")
st.write(f"Kokonaismatka: {total_distance:.2f} km")
st.write(f"Keskimääräinen askelpituus: {average_step_length:.2f} cm")

st.subheader("Kiihtyvyysdatan y-komponentti (suodatettu)")
downsampled_acceleration = acceleration_data['Linear Acceleration y (m/s^2)'].iloc[::10]
st.line_chart(downsampled_acceleration, use_container_width=True)

st.subheader("Tehospektri")
sampling_rate = 50
frequencies = np.fft.fftfreq(len(downsampled_acceleration), d=1/sampling_rate)
power_spectrum = np.abs(fft(downsampled_acceleration))**2
positive_freqs = (frequencies >= 0) & (frequencies <= 14)
chart_data = pd.DataFrame(np.transpose([frequencies[positive_freqs], power_spectrum[positive_freqs]]), columns=["Frequency (Hz)", "Power Spectrum"])

st.line_chart(chart_data, x="Frequency (Hz)", y="Power Spectrum", height=450)

st.subheader("Karttakuva")
start_lat = location_data['Latitude (°)'].mean()
start_long = location_data['Longitude (°)'].mean()
map = folium.Map(location=[start_lat, start_long], zoom_start=14)
folium.PolyLine(location_data[['Latitude (°)', 'Longitude (°)']], color='blue', weight=3.5, opacity=1).add_to(map)
st_folium(map, width=900, height=650)


# pip install pandas numpy streamlit scipy geopy folium streamlit-folium
# streamlit run SportsTracker.py