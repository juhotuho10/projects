package fi.tuni.prog3.weatherapp;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Scanner;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;



public class OpenWeatherMapAPI implements iAPI {

    private final String API_KEY;

    public OpenWeatherMapAPI() throws IOException {
        Path path = Paths.get("api_key.txt");
        if (!Files.exists(path)) {
            throw new IOException("API key file not found: " + path.toAbsolutePath());
        }
    this.API_KEY = Files.readString(path).trim();
}

    private String readApiKeyFromFile(String filename) throws IOException {
        Path path = Paths.get(filename);
        if (!Files.exists(path)) {
            throw new IOException("API key file not found: " + filename);
        }
        return Files.readString(path).trim();
    }

    // Error handling for city names
    private void validateCityName(String city) {
        if (city == null || city.trim().isEmpty()) {
            throw new IllegalArgumentException("City name cannot be null or empty");
        }
        if (!city.matches("^[\\p{L}\\s]+$")) {
            throw new IllegalArgumentException("City name must contain only letters and spaces");
        }
    }

    private JsonElement getApiJson(String ApiString) throws IOException {
        try {
            URL url = new URL(ApiString);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.connect();

            int responseCode = conn.getResponseCode();

            if (responseCode != 200) {
                throw new IOException("HttpResponseCode: " + responseCode);
            }

            StringBuilder informationString = new StringBuilder();
            Scanner scanner = new Scanner(url.openStream());

            while (scanner.hasNext()) {
                informationString.append(scanner.nextLine());
            }
            scanner.close();

            return JsonParser.parseString(informationString.toString());

        } catch (MalformedURLException e) {
            throw new IOException("URL is malformed: " + e.getMessage(), e);
        }
    }
    
    @Override
    public double[] lookUpLocation(String city) throws IOException {

        validateCityName(city);

        String urlString = String.format("http://api.openweathermap.org/geo/1.0/direct?q=%s&limit=1&appid=%s", city, API_KEY);

        double[] coordinates = new double[2];

        JsonObject jsonObject = getApiJson(urlString).getAsJsonArray().get(0).getAsJsonObject();

        if (jsonObject == null || jsonObject.isJsonNull()) {
            System.err.println("Error: Unable to find location for city: " + city);
            return new double[]{0.0, 0.0};
        }

        coordinates[0] = jsonObject.get("lat").getAsDouble();
        coordinates[1] = jsonObject.get("lon").getAsDouble();

        return coordinates;
    }

    @Override
    public JsonObject getCurrentWeather(String city) throws IOException {

        validateCityName(city);
        
        String urlString = String.format("http://api.openweathermap.org/data/2.5/weather?q=%s&appid=%s&units=metric", city, API_KEY);

        JsonObject jsonObject = getApiJson(urlString).getAsJsonObject();

        if (jsonObject == null || jsonObject.isJsonNull()) {
            System.err.println("Error: Unable to get current weather for city: " + city);
            return new JsonObject(); 
        }

        return jsonObject;
    }

    @Override
    public JsonArray getForecast(String city, Integer numDays) throws IOException {

        validateCityName(city);
        
        if (numDays == null || numDays < 1) {
            throw new IllegalArgumentException("Number of days must be 1 or higher");
        }

        double[] coords = lookUpLocation(city);
        
        String urlString = String.format("http://api.openweathermap.org/data/2.5/forecast/daily?lat=%f&lon=%f&cnt=%d&appid=%s&units=metric", coords[0], coords[1], numDays, API_KEY);

        JsonArray dataArray = getApiJson(urlString).getAsJsonObject().get("list").getAsJsonArray();

        if (dataArray == null || dataArray.isJsonNull()) {
            System.err.println("Error: Unable to get forecast for city: " + city);
            return new JsonArray(); 
        }

        return dataArray;
    }
}
