"""
A minesweeper game


Guide:
First the game opens up the configuration menu where you enter the minefield's
width and height in positive whole numbers. If you want, you can check
the recommended mine count based on the difficulty you want.
The mine count must be a positive whole number and it cannot be
more than there are buttons on the board.

After you have entered the minefield width, height and
mine count, you can start the game.

In the game itself, you  can left click the buttons to reveal them,
or right click unopened buttons in order to flag
them as potential mines. You cannot open flagged buttons.
if you suspect that you have wrongfully flagged a buttons
as dangerous, can remove the flag by right clicking the button again.

it's not necessary to flag buttons you suspect have mines in them,
but it can be a useful tool in keeping track of places where
you suspect mines might be.
it can also prevent accidentally opening buttons you didn't want to open

the numbers in the buttons tell you how many mines are around that
particular button, and the goal of the game is
to open all the buttons that do not have mines in them.

If you click on a mine, you lose and the game will
reveal all mine spots as red with the letter Ö (it looks like a mine)
Flagged mine spots will still show as neutral color F (as in Flag)

All the wrongfully flagged buttons will be marked with X

if you win the game, all the mines will show up with regular color and
all cleared tiles will show a green W symbol

There is also a quit button at the top left of the game,
if you want to close the program

"""

from tkinter import *
from tkinter import font
import random


class Input_Interface:

    def __init__(self):
        """
        defines self, the game input values and the text and
        the buttons of the option menu
        """
        self.start_game: bool = False
        self.width: int = 0
        self.height: int = 0
        self.mines_count: int = 0

        self.inputwindow = Tk()
        self.inputwindow.title("MineSweeper configuration")

        # ---------------- labels and entry --------------------------
        self.top_text = Label(self.inputwindow,
                              text="Minesweeper game\n "
                                   "input the minefield's width and height")

        self.width_text = Label(self.inputwindow,
                                text="Input the minefield width: (max 80)")
        self.width_input = Entry()

        self.height_text = Label(self.inputwindow,
                                 text="Input the minefield height: (max 50)")
        self.height_input = Entry()

        self.mine_text = Label(self.inputwindow, text="Input the mine count:")
        self.mine_input = Entry()

        self.recommendation_text = Label(self.inputwindow,
                                         text="Check the recommended mine count\n"
                                              "based on the minefield's width and height")

        self.recommended_mines_text = Label(self.inputwindow,
                                            text="Easy:   Normal:   hard:   Very hard:")

        self.recommendation_button = Button(self.inputwindow, text="Calculate",
                                            command=self.calculate_mines)

        # ---------------- buttons --------------------------

        self.start_button = Button(self.inputwindow, text="Start",
                                   command=self.start_program)

        self.quit_button = Button(self.inputwindow, text="Quit",
                                  command=self.quit_game)

        # ---------------- Grid placement --------------------------
        self.top_text.grid(row=0, column=1, sticky=N)

        self.width_text.grid(row=1, column=0, sticky=W)
        self.width_input.grid(row=1, column=2)

        self.height_text.grid(row=3, column=0, sticky=W)
        self.height_input.grid(row=3, column=2)

        self.mine_text.grid(row=7, column=0, sticky=W)
        self.mine_input.grid(row=7, column=2)

        self.recommendation_text.grid(row=4, column=0, sticky=W)
        self.recommended_mines_text.grid(row=5, column=0)

        self.recommendation_button.grid(row=6, column=0)

        self.start_button.grid(row=8, column=1, sticky=W)

        self.quit_button.grid(row=8, column=0, sticky=E)

    def start(self):
        """
        Starts the mainloop.
        """
        self.inputwindow.mainloop()

    def get_mines(self):
        """
        gets the mine input, checks that the input is fine and
        then assigns the input to self
        :return: bool, True or False
        """

        # gets values
        self.get_width_and_height()
        height = self.height
        width = self.width
        mines_count = self.mine_input.get()

        # checks if mines have a value at all
        if mines_count:
            # must be a number
            try:
                mines_count = int(mines_count)

                # must not be negative
                if mines_count >= 0:
                    # cannot be 0
                    if mines_count != 0:
                        # must be less than total button count
                        if mines_count < (width * height):
                            self.mines_count = mines_count
                            return True

                        else:
                            self.top_text.configure(
                                text="Error: cannot have more mines than tiles\n"
                                     "maximum mine count for this width and height is: "
                                     f"{(width * height) - 1}")
                            self.reset_mines()
                            return False

                    else:
                        self.top_text.configure(
                            text="Error: mines must not be 0")
                        self.reset_mines()
                        return False
                else:
                    self.top_text.configure(
                        text="Error: mines cannot be negative")
                    self.reset_mines()
                    return False

            except ValueError:
                self.top_text.configure(
                    text="Error: mines must be positive whole numbers.")
                self.reset_mines()
                return False
        else:
            self.top_text.configure(
                text="Error: mines cannot be empty! \n"
                     "Mines must be positive whole numbers.")
            return False

    def get_width_and_height(self):
        """
        gets the width and height input, checks that the input is fine and
        then assigns the input to self
        :return: bool, True or False
        """

        height = self.height_input.get()
        width = self.width_input.get()

        # checks if width and height have a value at all
        if width and height:
            # must be a number
            try:
                height = int(height)
                width = int(width)
                # must not be negative
                if width >= 0 and height >= 0:
                    # cannot be 0
                    if width != 0 or height != 0:
                        # must be under 60
                        if width <= 80 and height <= 50:

                            self.width = width
                            self.height = height
                            return True

                        else:
                            self.top_text.configure(
                                text="Error: height must not exceed 50\n"
                                     "and width must not exceed 80\n"
                                     "this is due to generation time being high")
                            self.reset_fields()
                            return False

                    else:
                        self.top_text.configure(
                            text="Error: height, width and mines must not be 0")
                        self.reset_fields()
                        return False
                else:
                    self.top_text.configure(
                        text="Error: height, width or mines cannot be negative")
                    self.reset_fields()
                    return False

            except ValueError:
                self.top_text.configure(
                    text="Error: height, width must be positive whole numbers.")
                self.reset_fields()
                return False
        else:
            self.top_text.configure(
                text="Error: height, width cannot be empty.\n"
                     "They must be positive whole numbers.")
            self.reset_fields()
            return False

    def calculate_mines(self):
        """
        gets the total buttons of the game and then suggests
        a mine count for estimated difficulty
        """

        if self.get_width_and_height():
            total_buttons = self.width * self.height

            easy = round(total_buttons * 0.10)
            normal = round(total_buttons * 0.14)
            hard = round(total_buttons * 0.18)
            very_hard = round(total_buttons * 0.22)

            # gives the recommended mine values and resets the top text
            self.recommended_mines_text.configure(
                text=f"Easy: {easy}  Normal:  {normal}  hard:  {hard} Very hard: {very_hard} ")

            self.top_text.configure(text="Minesweeper game\n"
                                         "input the minefield's width and height")

    def reset_fields(self):
        """
        In error situations this method will zeroes the elements:
        self.recommended_mines_text, self.height_input,
        self.width_input and self.mine_input.
        """

        self.height_input.delete(0, END)
        self.width_input.delete(0, END)
        self.mine_input.delete(0, END)

        self.recommended_mines_text.configure(
            text="Easy:     Normal:     hard:      Very hard:    ")

    def reset_mines(self):
        """
        In error situations this method will zeroes the elements:
        self.mine_input
        """
        self.mine_input.delete(0, END)

    def start_program(self):
        """
        stops the ui and continues to the game window with
        the parameters given by the user
        :return: bool, True
        """
        if self.get_width_and_height():
            if self.get_mines():
                self.start_game = True
                self.inputwindow.destroy()

    def quit_game(self):
        """
        quits the ui without starting the game
        """
        self.inputwindow.destroy()


class Game_Interface:

    def __init__(self, game_width: int, game_height: int, mines_count: int):
        """
        defines self, the quit button, the game parameters and
        coordinate lists and button dict
        """
        self.gamewindow = Tk()
        # background and title
        self.gamewindow.configure(bg="#d6d6d6")
        self.gamewindow.title("MineSweeper")

        # defines fonts
        self.font = font.Font(family="FreeSans", size=10, weight="bold")
        self.quit_font = font.Font(size=7)

        # defines quit button and places it in the grid
        self.quit_button = Button(self.gamewindow, text="Quit",
                                  font=self.quit_font,
                                  command=self.quit, height=1, width=2)
        self.quit_button.grid(row=0, column=0)

        self.width = game_width
        self.height = game_height
        self.mines_count = mines_count

        # all possible unopened coordinates
        self.all_possible_coordinates = []
        # all mine coordinates
        self.mine_coordinates = []

        # a dict that contains {coordinate: button object}
        # where coordinate is the button position in the grid
        self.button_dict = {}

        # generates buttons and puts them in the buttons dict,
        # also assigns coordinates in the all_possible_coordinates
        self.generate_buttons()
        # assigns mines to the buttons
        self.assign_mines()

    def start(self):
        """
        Starts the mainloop.
        """
        self.gamewindow.mainloop()

    def generate_buttons(self):
        """
        generates the button objects and assigns all of
        them coordinates based on the user input in the input menu
        """

        total_buttons = self.width * self.height

        # generates number list from 0 to width height times
        # for example width = 5 height = 2: [0,1,2,3,4,0,1,2,3,4]
        ButtonX = [i for _ in range(self.height) for i in range(self.width)]

        # generates number list enumerating the row as much as width
        # and as many times as height
        # for example width = 5 height = 2: [0,0,0,0,0,1,1,1,1,1]
        ButtonY = [i for i in range(self.height) for _ in range(self.width)]

        for i in range(total_buttons):
            # generates all buttons individually and
            # assigns coordinates from Button lists
            # for example width = 5 height = 2
            # (0,0 0,1 0,2.... 2,3, 2,4)
            button_coordinates = f"{ButtonY[i]},{ButtonX[i]}"

            # appends every coordinate to all possible coordinates
            self.all_possible_coordinates.append(button_coordinates)

            button = Button(self.gamewindow, text="", width=2, height=1,
                            relief=RAISED, borderwidth=1, bg="#d6d6d6",
                            font=self.font)

            # checks if the button is pressed with left or right click,
            # triggers different functions based on which button was pressed

            # lambda is used to create arguments for the
            # functions independently from bind, since bind itself doesn't
            # allow for arguments to be passed to the function called

            # the event (mouse button and position) is passed by bind,
            # the coordinates require definition
            # before being passed on to the function
            button.bind("<Button-1>",
                        lambda event, m=button_coordinates: self.press_command(
                            event, m))
            button.bind("<Button-3>",
                        lambda event, m=button_coordinates: self.flag_buttons(
                            event, m))

            # defines states for every button generated
            button.flagged = False
            button.opened = False

            # updates the dict with the {button_coordinates: button object}
            self.button_dict.update({button_coordinates: button})

            # assigns the button into grid based on the X and Y list values
            # buttonY has +1 to make room for the quit button
            button.grid(row=ButtonY[i] + 1, column=ButtonX[i])

    def assign_mines(self):
        """
        randomly assigns the given amount of mines to the buttons
        """

        # gets a copy of all the button coordinates
        # has to be a copy, otherwise we would be changing the original list
        available_minespots = self.all_possible_coordinates.copy()

        for i in range(self.mines_count):
            # random number between 0 and len(available_minespots)-1
            # from the 0:th element to the last element
            random_coordinate = random.randrange(len(available_minespots))

            # gets the mine spot from a randomly picked available spot
            minespot = available_minespots[random_coordinate]

            # deletes the chose coordinate from possible spots so
            # we can't double assign a mine
            available_minespots.pop(random_coordinate)
            self.mine_coordinates.append(minespot)

            selected_button = self.button_dict[minespot]
            # makes the button object have a mine value of True
            selected_button.has_mine = True

        # goes through all the unselected spots that
        # don't have a mine assigned to them
        for coordinate in available_minespots:
            selected_button = self.button_dict[coordinate]
            selected_button.has_mine = False

    def get_surrounding_coordinates(self, current_coordinates: str):
        """
        gets the surrounding coordinates of the
        button in question based on the buttons coordinates
        :param current_coordinates: the coordinates of the current button
        :return: returns a list of surrounding coordinates
        """

        # gets row and column values from a s name
        row, column = current_coordinates.split(",")
        row = int(row)
        column = int(column)

        surrounding_coordinates = []

        # [-1,-1 -1,0 -1,1]
        # [ 0,-1  0,0  0,1]
        # [ 1,-1  1,0  1,1]
        # gets the surrounding coordinates by adding
        # these numbers to the original coordinates

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                nearby_button = f"{row + i},{column + j}"

                # checks if the coordinate is even possible,
                # for example -1,-1 isn't a possible coordinate
                if nearby_button in self.all_possible_coordinates:

                    # excludes the current coordinate
                    # from surrounding coordinates
                    if nearby_button != current_coordinates:
                        surrounding_coordinates.append(nearby_button)

        return surrounding_coordinates

    def check_surrounding_mines(self, list_of_surroundings: list):
        """
        checks the mine count in the surrounding mines
        based on surrounding coordinates
        :param list_of_surroundings: a list of surrounding coordinates
        :return: int, a number of how many mines in the surrounding
        """
        total_mines = 0

        for coordinate in list_of_surroundings:
            # gets the button object from coordinate
            current_button = self.button_dict[coordinate]
            # check if the buttons has_mine == True
            if current_button.has_mine:
                total_mines += 1

        return total_mines

    def clear_surroundings(self, current_coordinates: str) -> list:
        """
        if button doesn't have any surrounding mines,
        the surrounding coordinates should be opened up and checked
        if they have mines in them

        function returns more coordinates to be cleared if
        the newly opened button doesn't have mines around it

        :param current_coordinates: the coordinates of the button with 0 mines
        :return: either a empty list,
                or a list of surrounding coordinates of the cleared button
        """

        current_button = self.button_dict[current_coordinates]

        # removes coordinates form all possible coordinates
        if current_coordinates in self.all_possible_coordinates:
            self.all_possible_coordinates.remove(current_coordinates)

        # doesn't open flagged buttons
        if not current_button.flagged:
            # declares the button to be opened
            current_button.opened = True
            # gets the surrounding coordinates
            surrounding_coordinates = self.get_surrounding_coordinates(
                current_coordinates)

            # amount of mines around
            mines_around = self.check_surrounding_mines(surrounding_coordinates)

            # colors assigned to the numbers based on
            # how many mines there are around the button (1-8)
            # colors are basically green, light yellow, orange, light red etc...
            colors = ["#00ff00", "#9fe800", "#e0af00", "#ad7600", "#9c3100",
                      "#690500", "#570017", "#2e0017"]
            # configures the button to look different when it has been opened up
            current_button.configure(bg="#bfbfbf",
                                     highlightbackground="#bfbfbf",
                                     activebackground="#bfbfbf")

            if mines_around == 0:
                current_button.configure(text="")
                # returns more coordinates to be opened up
                return surrounding_coordinates

            else:
                color = colors[mines_around - 1]
                # colors the number based on mines around
                current_button.configure(text=mines_around, fg=color,
                                         activeforeground=color)
                return []
        else:
            return []

    def press_command(self, event, current_coordinates):
        """
        command to open up the button that is clicked with left click,
        check if it has a mine and reveal how many
        mines there are in the surrounding coordinates,
        if no mines around, sends the surrounding buttons
        to be opened up too
        :param event: unused mouse button and coordinate values,
        simply carried over by the button bind command
        :param current_coordinates: the coordinates of the button in the button grid
        """
        current_button = self.button_dict[current_coordinates]

        # only opens up the button if the button isn't disabled and
        # if it hasn't been flagged by the user
        if current_button["state"] != "disabled" and not current_button.flagged:
            current_button.opened = True

            # makes the button clear queue list for buttons that need
            # to be opened in case the button you open has 0 mines around it
            button_clear_queue = []

            # append the clicked button coordinates to the clearing queue
            button_clear_queue.append(current_coordinates)

            # makes the button look different when opened
            current_button.configure(bg="#bfbfbf",
                                     highlightbackground="#bfbfbf",
                                     activebackground="#bfbfbf")

            # if you click on a mine, you lose
            if current_button.has_mine:
                self.lost_game()

            else:
                # goes through the whole clear queue before stopping
                while len(button_clear_queue) != 0:
                    # gets the first coordinate from the clear queue
                    coordinate = button_clear_queue[0]
                    # removes it so it doesn't get picked twice
                    button_clear_queue.pop(0)

                    # if mines around:
                    # assigns a number to the button and
                    # doesn't return new coordinates

                    # if no mines around:
                    # returns more button coordinates to be cleared
                    new_buttons = self.clear_surroundings(coordinate)

                    # if it isn't a possible coordinate, it get removed
                    # else it will be marked as opened
                    for coordinate in new_buttons:
                        if coordinate not in self.all_possible_coordinates:
                            new_buttons.remove(coordinate)
                        # removed flagged buttons
                        else:
                            self.all_possible_coordinates.remove(coordinate)

                    # the buttons are extended to the
                    # back of the queue to be opened
                    button_clear_queue.extend(new_buttons)

            # after opened button, check if the game has been won
            if self.check_win_condition():
                self.won_game()

    def won_game(self):
        """
        changes the color and text of the buttons and disables game input
        :return:
        """
        for button in self.button_dict.values():

            if button.has_mine:
                if not button.flagged:
                    button.configure(text="Ö")
            else:
                # all non mine buttons have W in them
                button.configure(text="W", bg="#71ff63")
        # disables button input
        # so you cannot change button states after the game ends
        self.disable_buttons()

    def lost_game(self):
        """
        reveals all the remaining mines and disables the game inputs
        also changes he color and text of the buttons
        """

        # for all the button objects
        for button in self.button_dict.values():

            # if button has been flagged by the user
            if button.flagged:
                # and the flag was wrongfully placed there
                if not button.has_mine:
                    button.configure(text="X")
            # if button not flagged
            else:
                # and button has a mine
                if button.has_mine:
                    button.configure(bg="#ff948c")
                    button.configure(text="Ö")

        self.disable_buttons()

    def check_win_condition(self):
        """
        checks if the all the unopened spots have mines in them
        :return: True or False based on if the game has been won
        """
        return set(self.mine_coordinates) == set(self.all_possible_coordinates)

    def disable_buttons(self):
        """
        disables buttons so that their states cannot be changed
        """
        for button in self.button_dict.values():
            button["state"] = DISABLED

    def flag_buttons(self, event, current_coordinates):
        """
        if the user right clicks, this function will flag the button
        :param event: unused mouse button and coordinate values,
                simply carried over by the button bind command
        :param current_coordinates: the coordinates of the button in the button grid
        """

        current_button = self.button_dict[current_coordinates]

        # if button isn't disabled or opened
        if not current_button.opened and current_button["state"] != "disabled":

            # if already flagged
            if current_button.flagged:
                # takes the flag off,
                # the foreground color doesnt need to be reverted,
                # since there isn't anything that would be colored by it
                current_button.configure(text="")
                current_button.flagged = False

            # if not flagged
            elif not current_button.flagged:
                # places a F symbol for flag and changed foreground color
                current_button.configure(text="F", fg="#ff6363")
                current_button.flagged = True

    def quit(self):
        """
        quits from the game and closes the window
        """
        self.gamewindow.destroy()


def main():
    """
    starts the config menu and the game itself
    """
    ui = Input_Interface()
    ui.start()

    # gets the user input values from the options menu
    width, height, mines_count, start = ui.width, ui.height, ui.mines_count, ui.start_game

    # only if the user has started the game,
    # so the game doesn't run, if you just click X on the menu
    if start:
        game = Game_Interface(width, height, mines_count)
        game.start()


if __name__ == '__main__':
    main()
