package fi.tuni.prog3.weatherapp;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.ArrayList;

public class FileHandler implements iReadAndWriteToFile {

    @Override
    public ArrayList<String> readFromFile(String fileName) throws IOException {
        Gson gson = new Gson();
        Type listType = new TypeToken<ArrayList<String>>() {}.getType();
        ArrayList<String> data;

        try (FileReader reader = new FileReader(fileName)) {
            data = gson.fromJson(reader, listType);
            System.out.println("Data loaded from " + fileName);
        }

        return data;
    }

    @Override
    public void writeToFile(String fileName, ArrayList<String> data) throws IOException {
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String json = gson.toJson(data);

        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write(json);
            System.out.println("Data saved to " + fileName);
        }
    }
}
