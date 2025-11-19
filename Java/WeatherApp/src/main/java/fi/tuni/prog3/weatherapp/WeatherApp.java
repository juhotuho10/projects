package fi.tuni.prog3.weatherapp;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.control.ScrollPane;
import javafx.scene.transform.Rotate;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.control.ListView;

import javafx.event.EventHandler;
import javafx.scene.input.MouseEvent;

import com.google.gson.JsonObject;

import com.google.gson.JsonArray;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.io.*;
import java.util.ArrayList;



/**
 * A JavaFX application that displays weather information.
 * It interacts with OpenWeatherMapAPI to fetch current and forecast weather data.
 */
public class WeatherApp extends Application {

    private static int SCENE_WIDTH = 500;
    private static int SCENE_HEIGTH = 700;

    private Label CityLabel = new Label("");
    private Pane centerPane = new Pane();
    private Pane mainPane = new Pane();
    private ScrollPane scrollPane = new ScrollPane();
    private HBox smallPaneContainer = new HBox(10);
    ListView<String> historyListView = new ListView<>(); // Create ListView
    private ArrayList<String> searchHistory = new ArrayList<>(); // Store recent searches
    private OpenWeatherMapAPI api;

    public WeatherApp() {
        try {
            this.api = new OpenWeatherMapAPI();
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("Failed to read API key");
            Platform.exit();
        }
    }

    private FileHandler filehandler = new FileHandler();

    /**
     * Initializes and displays the primary stage of the application.
     * @param stage The primary stage for this application.
     * @throws IOException If there is an issue reading the search history file.
     */
    @Override
    public void start(Stage stage) throws IOException{

        this.searchHistory = filehandler.readFromFile("search_history.json");
        
        BorderPane root = new BorderPane();
        root.setPadding(new Insets(10, 10, 10, 10));
        
        root.setCenter(getCenterPane());
        
        var quitButton = getQuitButton();
        BorderPane.setMargin(quitButton, new Insets(10, 10, 0, 10));
        root.setBottom(quitButton);
        BorderPane.setAlignment(quitButton, Pos.TOP_RIGHT);
        
        Scene scene = new Scene(root, SCENE_WIDTH, SCENE_HEIGTH);                      
        stage.setScene(scene);
        stage.setTitle("WeatherApp");
        stage.show();
        stage.setResizable(false);

        scene.addEventFilter(MouseEvent.MOUSE_CLICKED, new EventHandler<MouseEvent>() {
            @Override
            public void handle(MouseEvent mouseEvent) {
                if (historyListView.isVisible()) {
                    historyListView.setVisible(false);
                }
            }
        });

        if (this.searchHistory.size() > 0 ){
            fetchCityDataFromInput(searchHistory.get(0));
        }else{
            fetchCityDataFromInput("Tampere");
        }
    }
    
    /**
     * Invoked when the application should stop, and provides a chance to clean up resources.
     * Here, it's used to save the search history.
     * @throws IOException If there is an issue writing the search history file.
     */
    @Override
    public void stop() throws IOException{
        // Save history when the app closes
        filehandler.writeToFile("search_history.json", searchHistory);
    }

    /**
     * The main entry point for all JavaFX applications.
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        launch();
    }
    
    /**
     * Creates and returns the central pane of the application's GUI.
     * This pane includes a text field for city input, a search button, and areas to display weather information.
     * @return The central pane of the application.
     */
    private Pane getCenterPane() {

        centerPane.getChildren().clear();

        historyListView.setPrefSize(150, 200);
        historyListView.getItems().setAll(searchHistory);
        historyListView.setVisible(false);

        // Add a selection listener to the historyListView
        historyListView.getSelectionModel().selectedItemProperty().addListener((observable, oldValue, newValue) -> {
        if (newValue != null) {
            fetchCityDataFromInput(newValue);
            historyListView.setVisible(false);
        }
        });

        // Add components to the pane
        TextField textField = new TextField();
        textField.setPromptText("Enter city name");
        textField.setOnMouseClicked(e -> historyListView.setVisible(!historyListView.isVisible()));
    
    
        Button fetchButton = new Button("Search"); 

        fetchButton.setOnAction(e -> {
            try {
                String location = textField.getText().trim();
                validateCityName(location);
                fetchCityDataFromInput(location);
                addToSearchHistory(location);
            } catch (IllegalArgumentException ex) {
                displayError(ex.getMessage());
            }
        });
        
        centerPane.setStyle("-fx-background-color: #8fc6fd;");
    
        CityLabel.setFont(Font.font("Arial", FontWeight.BOLD, 20));

        smallPaneContainer.setPadding(new Insets(5));
        scrollPane.setContent(smallPaneContainer);
        scrollPane.setFitToHeight(true);
        scrollPane.setPrefSize(SCENE_WIDTH - 20, 310);

        Label ForecastLabel = new Label("Weather forecast:");
        ForecastLabel.setFont(Font.font("Arial", FontWeight.BOLD, 15));

        // Positioning components
        CityLabel.relocate(20, 10);
        textField.relocate(330, 10);
        fetchButton.relocate(280, 10);
        ForecastLabel.relocate(10, 280);
        scrollPane.relocate(0, 300);
        historyListView.relocate(330,35);        

        centerPane.getChildren().addAll(CityLabel, textField, fetchButton, this.mainPane, scrollPane, historyListView, ForecastLabel);
    
        return centerPane;
    }

    /**
     * Creates and returns a Pane displaying the current weather of a city.
     * @param cityJSON The JSON object containing the city's weather data.
     * @return A Pane displaying the city's current weather.
     */
    private Pane makeMainPane(JsonObject cityJSON){

        Pane tempPane = new Pane();
        tempPane.relocate(0, 50); 
        tempPane.setPrefSize(SCENE_WIDTH - 20, 200); 
        tempPane.setStyle("-fx-background-color: ADD8E6;");

        String tempString = getTempFromJson(cityJSON);
        Label temperatureLabel = new Label("Temperature: " + tempString + "°C");

        String descriptionString = getDescriptionFromJson(cityJSON);
        Label descriptionLabel = new Label(descriptionString);

        ImageView weatherImageView = makeImgBox(100);
        String imageUrl = getImageUrlFromJson(cityJSON);
        Image image = new Image(imageUrl);
        weatherImageView.setImage(image);

        Double[] SpeedRotation = getWindFromJson(cityJSON);

        Label WindLabel = new Label("Wind: " + SpeedRotation[0] + "m/s");

        ImageView WindImg = makeWindArrow(SpeedRotation[1], 40, 310, 140);
        
        temperatureLabel.relocate(200, 10);
        descriptionLabel.relocate(200, 25);
        weatherImageView.relocate(200, 30);
        WindLabel.relocate(200, 130);
        
        
        tempPane.getChildren().addAll(temperatureLabel, descriptionLabel, weatherImageView, WindLabel, WindImg);

        return tempPane;

    }

    /**
     * Adds a new location to the beginning of the search history.
     * If the history exceeds 10 entries, it trims the list.
     * @param location The city location to add to the search history.
     */
    private void addToSearchHistory(String location) {
        searchHistory.add(0, location);

        if  (this.searchHistory.size() > 10){
            this.searchHistory = new ArrayList<>(searchHistory.subList(0, 10));
        }

        historyListView.getItems().setAll(searchHistory);
    }

    /**
     * Creates and returns an ImageView for displaying images, such as weather icons.
     * @param size The desired size of the ImageView.
     * @return A configured ImageView.
     */
    private ImageView makeImgBox(int size) {
        ImageView imageView = new ImageView();
        imageView.setFitHeight(size); 
        imageView.setFitWidth(size);  
        imageView.setPreserveRatio(true);
        return imageView;
    }
    
    /**
     * Creates and returns a "Quit" button that terminates the application.
     * @return A Button labeled "Quit".
     */
    private Button getQuitButton() {
        Button button = new Button("Quit");
        
        //Adding an event to the button to terminate the application.
        button.setOnAction((ActionEvent event) -> {
            Platform.exit();
        });
        
        return button;
    }

    /**
     * Creates a small pane for displaying weather forecast information.
     * @param predictionJson The JSON object containing the forecast data.
     * @return A Pane displaying the forecast information.
     */
    private Pane makeSmallPane(JsonObject predictionJson){
        Pane smallPane = new Pane();
        smallPane.setPrefSize(120, 180); 
        smallPane.setStyle("-fx-background-color: #D3D3D3;");

        Integer unixTime = getUnixTimeFromJson(predictionJson);

        // dd-MM-yyyy HH:mm:ss
        String[] datetime = UnixToDatetime(unixTime).split(" ");

        Label DateLabel = new Label(datetime[0]);
        
        Label TempLabel = new Label("Temperature: ");
        
        String[] MinMax = getTempFromPrediction(predictionJson);
        Label CelciusLabel = new Label(MinMax[0] + "°C to " + MinMax[1] + "°C");

        String WeatherImgUrl = getImageUrlFromJson(predictionJson);
        Image Weatherimage = new Image(WeatherImgUrl);

        ImageView weatherImg = makeImgBox(60);

        weatherImg.setImage(Weatherimage);

        Double[] SpeedRotation = getWindFromPrediction(predictionJson);

        Label WindLabel = new Label("Wind: " + SpeedRotation[0] + "m/s");
        
        ImageView WindImg = makeWindArrow(SpeedRotation[1], 20, 60, 140);

        String rainProb = getRainProbFromJson(predictionJson);
        Label RainProbLabel = new Label("Rain prob: " + rainProb + "%");
        
        DateLabel.relocate(30, 5);
        TempLabel.relocate(25, 30);
        CelciusLabel.relocate(12, 45);
        weatherImg.relocate(30, 50);
        WindLabel.relocate(20, 100);
        RainProbLabel.relocate(20, 160);
       
        smallPane.getChildren().addAll(DateLabel, TempLabel, CelciusLabel, weatherImg, WindLabel, WindImg, RainProbLabel);
    
        return smallPane;
    }

    /**
     * Sets the city label's text to the provided city name, formatted with capitalization.
     * @param city The city name to set on the label.
     */
    private void setCityLabel(String city){
        String s1 = city.substring(0, 1).toUpperCase();
        String cityCapitalized = s1 + city.substring(1).toLowerCase();
        CityLabel.setText(cityCapitalized);
    }

    /**
     * Fetches and displays weather data for the provided city.
     * @param city The city for which to fetch and display weather data.
     */
    private void fetchCityDataFromInput(String city) {

        try {
            if (city != null && !city.trim().isEmpty()) {
                JsonObject cityJSON = api.getCurrentWeather(city);

                if (cityJSON != null) {

                    this.mainPane = makeMainPane(cityJSON);

                    centerPane.getChildren().remove(this.mainPane);
                    centerPane.getChildren().add(this.mainPane);

                    refreshSmallPanes(city);
                    setCityLabel(city);

                    historyListView.toFront();

                } else {
                    displayError("No weather data available for " + city);
                }
            } else {
                displayError("City name is empty.");
            }
        } catch (IOException | IllegalArgumentException e) {
            displayError(e.getMessage());
        }
    }


    /**
     * Clears existing forecast panes and populates them with new data for the given city.
     * @param city The city for which to refresh the forecast panes.
     */
    private void refreshSmallPanes(String city){
        try {
            smallPaneContainer.getChildren().clear();

            JsonArray predArray = api.getForecast(city, 10);

            for (int i = 0; i < predArray.size(); i++) {
                JsonObject days_prediction = predArray.get(i).getAsJsonObject();
                Pane smallPane = makeSmallPane(days_prediction);
                smallPaneContainer.getChildren().add(smallPane);
            }
        } catch (IOException | IllegalArgumentException e) {
            displayError(e.getMessage());
        }
    }

    /**
     * Creates and returns an ImageView of a wind arrow, rotated according to the provided wind direction.
     * @param rotation The rotation angle for the wind direction.
     * @param arrow_size The size of the wind arrow image.
     * @param x The x-coordinate for positioning the image.
     * @param y The y-coordinate for positioning the image.
     * @return An ImageView representing the wind direction.
     */
    private ImageView makeWindArrow(Double rotation, Integer arrow_size, Integer x, Integer y){

        ImageView WindImg = makeImgBox(arrow_size);
        double middlePoint = arrow_size/2;
        Rotate rotate = new Rotate();
        rotate.setPivotX(middlePoint);  
        rotate.setPivotY(middlePoint);  
        rotate.setAngle(rotation);   
        WindImg.getTransforms().add(rotate);   

        String WindImgUrl = "https://upload.wikimedia.org/wikipedia/commons/6/61/Black_Up_Arrow.png";
        Image Windimage = new Image(WindImgUrl);
        WindImg.setImage(Windimage);
        WindImg.relocate(x - middlePoint, y - middlePoint);

        return WindImg;

    };

    /**
     * Converts a UNIX timestamp into a human-readable date and time format.
     * @param unixTime The UNIX timestamp.
     * @return A string representation of the date and time.
     */
    private String UnixToDatetime(Integer unixTime){

        LocalDateTime dateTime = LocalDateTime.ofInstant(Instant.ofEpochSecond(unixTime), ZoneId.systemDefault());

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd.MM.yyyy HH:mm:ss");

        return dateTime.format(formatter);
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

    /**
     * Displays an error message in a popup dialog.
     * @param message The error message to display.
     */
    private void displayError(String message) {
        Alert alert = new Alert(AlertType.ERROR);
        alert.setTitle("Error");
        alert.setHeaderText(null); // No header text
        alert.setContentText(message);

        alert.showAndWait();
    }

    // Collection of helper fuctions for parsin JsonObjects and getting relevant data out of them

    private String getImageUrlFromJson(JsonObject data){
        JsonArray weatherData = data.getAsJsonArray("weather");
        String icon = weatherData.get(0).getAsJsonObject().get("icon").getAsString();
        String imageUrl = "https://openweathermap.org/img/wn/" + icon + "@2x.png";
        return imageUrl;

    }

    private String getTempFromJson(JsonObject data){
        String tempString = data.getAsJsonObject("main").get("temp").getAsString();
        return tempString;

    }

    private String[] getTempFromPrediction(JsonObject data){
        String[] MinMax = new String[2];
        MinMax[0] = data.getAsJsonObject("temp").get("min").getAsString();
        MinMax[1] = data.getAsJsonObject("temp").get("max").getAsString();
        return MinMax;

    }

    private int getUnixTimeFromJson(JsonObject data){
        int UnixTime = (int)data.get("dt").getAsDouble();
        return UnixTime;

    }

    private Double[] getWindFromJson(JsonObject data){
        Double[] SpeedRotation = new Double[2];
        JsonObject WindJson = data.getAsJsonObject("wind");
        SpeedRotation[0] = WindJson.get("speed").getAsDouble();
        SpeedRotation[1] = WindJson.get("deg").getAsDouble();
        return SpeedRotation;

    }

    private Double[] getWindFromPrediction(JsonObject data){
        Double[] SpeedRotation = new Double[2];
        SpeedRotation[0] = data.get("speed").getAsDouble();
        SpeedRotation[1] = data.get("deg").getAsDouble();
        return SpeedRotation;

    }

    private String getRainProbFromJson(JsonObject data){
        double rainProbDouble = data.get("pop").getAsDouble() * 100;
        int rainProbInt = (int)rainProbDouble;
        
        return Integer.toString(rainProbInt);

    }

    private String getDescriptionFromJson(JsonObject data){
        JsonArray weatherData = data.getAsJsonArray("weather");
        String descriptionString = weatherData.get(0).getAsJsonObject().get("description").getAsString();
        return descriptionString;

    }
} 