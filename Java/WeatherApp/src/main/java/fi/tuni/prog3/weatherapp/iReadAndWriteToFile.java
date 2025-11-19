/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Interface.java to edit this template
 */
package fi.tuni.prog3.weatherapp;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Interface with methods to read from and write to a file.
 */
public interface iReadAndWriteToFile {

    /**
     * Reads an ArrayList<String> from the given file.
     * @param fileName Name of the file to read from.
     * @return ArrayList<String> read from the file.
     * @throws IOException if there is an issue reading the file.
     */
    ArrayList<String> readFromFile(String fileName) throws IOException;

    /**
     * Writes an ArrayList<String> to the given JSON file.
     * @param fileName Name of the file to write to.
     * @param data ArrayList<String> to be written into the file.
     * @throws IOException if there is an issue writing to the file.
     */
    void writeToFile(String fileName, ArrayList<String> data) throws IOException;
}
