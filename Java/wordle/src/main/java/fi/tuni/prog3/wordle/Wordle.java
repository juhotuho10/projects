package fi.tuni.prog3.wordle;

 

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.security.interfaces.RSAKey;
import java.util.ArrayList;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.Stage;


public class Wordle extends Application {

    // Array to store the words
    private ArrayList<String> words = new ArrayList<>();
    private String targetWord;
    private ArrayList<Character> CurrGuess = new ArrayList<>();
    private int guessCount = 0;

    private int MAX_WORD_LENGTH = 5; // Default word length
    private static final int MAX_ATTEMPTS = 6; 
    private TextField[][] textFields;
    private GridPane grid = new GridPane();
    private Button ResetButton = new Button();
    private Text TextBox = new Text("");
    private VBox layout = new VBox(0);
    private HBox topbox = new HBox(10);
    private Boolean gameDone = false;
    private int wordIndex = 0;

    // main initialization
    @Override
    public void start(Stage stage) {
        this.words = load_file("words.txt");
        initializebutton();
        initialize_ui();
        Scene scene = new Scene(layout, 600, 600);
        stage.setTitle("Wordle");
        stage.setScene(scene);
        stage.show();

    }

    // makes a i by j grid of textfields as a form to place to input text
    private void initializeGrid(int Word_len) {

        for (int i = 0; i < MAX_ATTEMPTS; i++) {
            for (int j = 0; j < Word_len; j++) {
                TextField textField = new TextField();
                textField.setPrefWidth(60); // Set preferred width to make it look like a square
                textField.setPrefHeight(60); // Set preferred height to make it look like a square
                textField.setMaxWidth(60);
                textField.setMaxHeight(60);
                textField.setEditable(false); // control the input via key events, not direct typing
                textField.setText("");
                textField.setStyle("-fx-control-inner-background: white;");
                String s = String.format("%d_%d", i,j);

                textField.setId(s);

                //  allow a single character in the text field
                textField.textProperty().addListener((observable, oldValue, newValue) -> {
                    if (newValue.length() > 1) {
                        textField.setText(newValue.substring(0, 1));
                    }
                });

                textFields[i][j] = textField;
                grid.add(textField, j, i); 
            }
        }

        // Set focus traversal keys to none, so the focus doesn't move automatically on tab or other keys
        grid.setFocusTraversable(false);

        // handle all key presses and direct them to the appropriate cell
        grid.addEventFilter(KeyEvent.KEY_PRESSED, event -> {
            if (this.gameDone){
                return;
            }
            if (event.getCode().isLetterKey()) {
                for (int i = 0; i < MAX_ATTEMPTS; i++) {
                    for (int j = 0; j < MAX_WORD_LENGTH; j++) {
                        TextField currentField = textFields[i][j];
                        if (currentField.getText().equals("") && i == this.guessCount) {
        
                            currentField.setText(event.getText().toUpperCase());
      
                            char key_press = currentField.getText().toUpperCase().charAt(0);
                            this.CurrGuess.set(j, key_press);
        
                            int nextCol = (j + 1) % MAX_WORD_LENGTH;
                            int nextRow = i;
      
                            if (nextRow <= MAX_ATTEMPTS) {
                                textFields[nextRow][nextCol].requestFocus();
                            }
                            break;
                        }
                    }
                }
                 
            }else if (event.getCode() == KeyCode.ENTER){
                // Use enter key for checking the guess and initiating the coloring of the cells
                this.TextBox.setText("");
                if (check_full_row()){
                    ArrayList<Integer> correct_guess = check_correct_chars();
                    //System.out.println(correct_guess);
                    for (int i = 0; i < MAX_WORD_LENGTH; i++) {
                        TextField currentField = textFields[guessCount][i];
                        int correct = correct_guess.get(i);
                        if (correct == 0){
                            currentField.setStyle("-fx-control-inner-background: grey;-fx-text-fill: white;");
                        }else if (correct == 1){
                            currentField.setStyle("-fx-control-inner-background: orange;-fx-text-fill: white;");
                        }else{
                            currentField.setStyle("-fx-control-inner-background: green;-fx-text-fill: white;");
                        }
                    }

                    if (check_guess()){ 
                        this.TextBox.setText("Congratulations, you won!");
                        this.gameDone = true;
                    }else if(this.guessCount + 1 == MAX_ATTEMPTS) {
                        this.TextBox.setText("Game over, you lost!");
                        this.gameDone = true;
                    }else{
                        this.guessCount += 1;
                        reset_guess();
                    }

                }else{
                    this.TextBox.setText("Give a complete word before pressing Enter!");
                }

            
            }else if (event.getCode() == KeyCode.BACK_SPACE){
                // backspace will clear the most recent cell that was writte on 
                for (int i = MAX_WORD_LENGTH-1; i >= 0; i--) {
                        TextField currentField = textFields[guessCount][i];
                        if (!currentField.getText().equals("")){
                            currentField.setText("");
                            this.CurrGuess.set(i, '_');
                            break;
                        }
                    }
            }
            // Consume the event so it doesn't propagate further
            event.consume();
            }
        );
    }

    // initializes a new UI and increments the word to guess
    private void reset_grid(){

        this.wordIndex += 1;

        initialize_ui();
        
    }

    // clear the previous UI and initializes a new one according to the current word
    private void initialize_ui(){
        this.TextBox.setId("infoBox");

        this.targetWord = this.words.get(this.wordIndex);
        this.MAX_WORD_LENGTH = this.targetWord.length();

        this.textFields = new TextField[MAX_ATTEMPTS][this.MAX_WORD_LENGTH];

        this.grid = new GridPane();
        
        initializeGrid(this.MAX_WORD_LENGTH);

        layout.getChildren().clear();

        topbox.getChildren().clear();

        this.topbox.getChildren().addAll(ResetButton, TextBox);

        layout.getChildren().addAll(this.topbox, this.grid);

        this.CurrGuess = new ArrayList<>();

        for (int i = 0; i < this.MAX_WORD_LENGTH; i++) {
            this.CurrGuess.add('_');
        }

        this.textFields[0][0].requestFocus();
        

    }

    // makes and defines the new game button
    private void initializebutton() {
        
        ResetButton.setText("Start new game");
        ResetButton.setOnAction(event -> {
            for (int i = 0; i < MAX_ATTEMPTS; i++) {
                for (int j = 0; j < MAX_WORD_LENGTH; j++){
                    TextField currentField = textFields[i][j];
                    currentField.setText("");
                    currentField.setStyle("--fx-control-inner-background: white;-fx-text-fill: black;");
                }
            }
            this.gameDone = false;
            reset_guess();
            reset_grid();
            this.guessCount = 0;
            this.TextBox.setText("");
            this.textFields[0][0].requestFocus();
        });

        ResetButton.setId("newGameBtn");
    }

    // loads the text file into a arraylist so we can access the words easier
    private ArrayList<String> load_file(String filePath){
        ArrayList<String> words = new ArrayList<>();

        try {
            FileReader fileReader = new FileReader(filePath);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            String line;

            while ((line = bufferedReader.readLine()) != null) {
                words.add(line.toUpperCase());
            }
            bufferedReader.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

        return words;

    }

    // checks if the current row has the correct guess
    private Boolean check_guess(){

        StringBuilder builder = new StringBuilder();
        for (Character CurrChar : this.CurrGuess) {
            builder.append(CurrChar);
        }

        String str = builder.toString();

        return this.targetWord.equals(str);
    }

    // checks which chars are correct, 0 = wrong char. 1 = right char, wrong place, 2 = correct char correct place
    // and returns a array list with those integers
    private ArrayList<Integer> check_correct_chars(){

        ArrayList<Integer> correct_guess = new ArrayList<>();

        for (Character CurrChar : this.CurrGuess) {
            if (this.targetWord.indexOf(CurrChar) != -1){
                correct_guess.add(1);
            }else{
                correct_guess.add(0);
            }
        }

        for (int i = 0; i < this.MAX_WORD_LENGTH; i++) {
            char correct = this.targetWord.charAt(i);
            char guess = this.CurrGuess.get(i);
            if (correct == guess){
                correct_guess.set(i, 2);
            }
        }
        return correct_guess;
       
    }

    // checks if all the rows letters have something written on them
    private Boolean check_full_row(){
        for (int i = 0; i < this.MAX_WORD_LENGTH; i++) {
            if (this.CurrGuess.get(i) == '_'){
                return false;
            }
        }

        return true;
    }

    // resets the current guess
    private void reset_guess(){
        for (int i = 0; i < this.MAX_WORD_LENGTH; i++) {
            this.CurrGuess.set(i, '_');
        }
    }

    public static void main(String[] args) {
        launch();
    }

}