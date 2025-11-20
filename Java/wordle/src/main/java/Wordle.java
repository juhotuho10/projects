
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
    private String targetWord; // The word to be guessed
    private ArrayList<Character> CurrGuess = new ArrayList<>();
    private int guessCount = 0;

    private int MAX_WORD_LENGTH = 5; // Default word length
    private static final int MAX_ATTEMPTS = 6; // Number of attempts typically allowed in Wordle
    private TextField[][] textFields;
    private GridPane grid = new GridPane();
    private Button ResetButton = new Button();
    private Text TextBox = new Text("");
    private VBox layout = new VBox(0);
    private HBox topbox = new HBox(10);
    private Boolean gameDone = false;
    private int wordIndex = 0;

    @Override
    public void start(Stage stage) {
        this.TextBox.setId("infoBox");
        this.words = load_file("words.txt");
        this.targetWord = this.words.get(this.wordIndex);
        //System.out.println(this.targetWord);
        this.MAX_WORD_LENGTH = this.targetWord.length();
        
        for (int i = 0; i < this.MAX_WORD_LENGTH; i++) {
            this.CurrGuess.add('_');
        }

        this.textFields = new TextField[MAX_ATTEMPTS][MAX_WORD_LENGTH];
        initializebutton();
        initializeGrid(MAX_WORD_LENGTH);

        topbox.getChildren().addAll(ResetButton, TextBox);

        layout.getChildren().addAll(topbox, grid);

        Scene scene = new Scene(layout, 500, 500);
        stage.setTitle("Wordle");
        stage.setScene(scene);
        stage.show();

        textFields[0][0].requestFocus();
    }

    private void initializeGrid(int Word_len) {

        //System.out.println(Word_len);
        for (int i = 0; i < MAX_ATTEMPTS; i++) {
            for (int j = 0; j < Word_len; j++) {
                TextField textField = new TextField();
                textField.setPrefWidth(60); // Set preferred width to make it look like a square
                textField.setPrefHeight(60); // Set preferred height to make it look like a square
                textField.setMaxWidth(60); // Ensure that it doesn't grow larger than this size
                textField.setMaxHeight(60);
                textField.setEditable(false); // We'll control the input via key events, not direct typing
                textField.setText("");
                textField.setStyle("-fx-control-inner-background: white;");
                String s = String.format("%d_%d", i,j);

                textField.setId(s);

                // Only allow a single character in the text field
                textField.textProperty().addListener((observable, oldValue, newValue) -> {
                    if (newValue.length() > 1) {
                        textField.setText(newValue.substring(0, 1));
                    }
                });

                textFields[i][j] = textField;
                grid.add(textField, j, i); // Add to grid (col, row)
            }
        }

        // Set focus traversal keys to none, so the focus doesn't move automatically on tab or other keys
        grid.setFocusTraversable(false);

        // This event filter will handle all key presses and direct them to the appropriate cell
        grid.addEventFilter(KeyEvent.KEY_PRESSED, event -> {
            if (this.gameDone){
                return;
            }
            if (event.getCode().isLetterKey()) {
                for (int i = 0; i < MAX_ATTEMPTS; i++) {
                    for (int j = 0; j < MAX_WORD_LENGTH; j++) {
                        TextField currentField = textFields[i][j];
                        if (currentField.getText().equals("") && i == this.guessCount) {
                            // Ensure we're handling letter keys only
                            currentField.setText(event.getText().toUpperCase());
                            // Move focus to next field, wrap around at the end of the row
                            char key_press = currentField.getText().toUpperCase().charAt(0);
                            this.CurrGuess.set(j, key_press);
                            //System.out.println(this.CurrGuess);
                            int nextCol = (j + 1) % MAX_WORD_LENGTH;
                            int nextRow = i;
                            //System.out.println(check_guess());
                            //System.out.println();

                            //System.out.println(nextCol);
                            // Prevent ArrayIndexOutOfBoundsException for the last element
                            if (nextRow <= MAX_ATTEMPTS) {
                                textFields[nextRow][nextCol].requestFocus();
                            }
                            break;
                        }
                    }
                }
                 // Consume the event so it doesn't propagate further
            }else if (event.getCode() == KeyCode.ENTER){
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
                for (int i = MAX_WORD_LENGTH-1; i >= 0; i--) {
                        TextField currentField = textFields[guessCount][i];
                        if (!currentField.getText().equals("")){
                            currentField.setText("");
                            this.CurrGuess.set(i, '_');
                            break;
                        }
                    }
            }

            event.consume();
            }
        );
    }

    private void reset_grid(){

        this.wordIndex += 1;

        this.targetWord = this.words.get(this.wordIndex);
        this.MAX_WORD_LENGTH = this.targetWord.length();

        this.textFields = new TextField[MAX_ATTEMPTS][this.MAX_WORD_LENGTH];

        this.grid = new GridPane();
        
        initializeGrid(this.MAX_WORD_LENGTH);

        layout.getChildren().clear();

        layout.getChildren().addAll(this.topbox, this.grid);

        this.CurrGuess = new ArrayList<>();

        for (int i = 0; i < this.MAX_WORD_LENGTH; i++) {
            this.CurrGuess.add('_');
        }

        this.textFields[0][0].requestFocus();
        
    }

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

    private Boolean check_guess(){

        StringBuilder builder = new StringBuilder();
        for (Character CurrChar : this.CurrGuess) {
            builder.append(CurrChar);
        }

        String str = builder.toString();

        return this.targetWord.equals(str);
    }

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

    private Boolean check_full_row(){
        for (int i = 0; i < this.MAX_WORD_LENGTH; i++) {
            if (this.CurrGuess.get(i) == '_'){
                return false;
            }
        }

        return true;
    }

    private void reset_guess(){
        for (int i = 0; i < this.MAX_WORD_LENGTH; i++) {
            this.CurrGuess.set(i, '_');
        }
    }

    public static void main(String[] args) {
        launch();
    }

}