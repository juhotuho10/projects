/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Interface.java to edit this template
 */

package fi.tuni.prog3.weatherapp;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.IOException;

/**
 * Interface with methods to interact with the openweathermap API.
 */
public interface iAPI {
    
    /**
     * Returns coordinates for a given city.
     * @param city The city name.
     * @return double[]
     */
    double[] lookUpLocation(String city) throws IOException, IllegalArgumentException;
    
    /**
     * Returns the current weather for the given city.
     * @param city The city name.
     * @return JsonObject
     */
    JsonObject getCurrentWeather(String city) throws IOException, IllegalArgumentException;

    /**
     * Fetches a prediction array from the API for a given city for numDays in the future.
     * @param city The city name.
     * @param numDays Number of days for the forecast.
     * @return JsonArray.
     */
    JsonArray getForecast(String city, Integer numDays) throws IOException, IllegalArgumentException;
}
