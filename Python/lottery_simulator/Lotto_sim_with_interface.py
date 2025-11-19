import random
import time
import pygame

pygame.init()

Display_Width = 800
Display_Height = 600

keys = pygame.key.get_pressed()
clock = pygame.time.Clock()
smallertext = pygame.font.SysFont('ariali.ttf', 25)
smallText = pygame.font.SysFont('ariali.ttf', 30)
bigtext = pygame.font.SysFont('ariali.ttf', 40)

speed = 100
Jackpots = 0
TotalWins = 0
YearsRun = 0
WeeksRun = 0

# RGB värien määritys (255 on max)
White = (210, 210, 210)
Grey = (50, 50, 45)
DarkGrey = (35, 35, 30)
LightGrey = (80, 80, 75)
Green = (0, 210, 0)
DarkGreen = (0, 100, 0)
Blue = (0, 0, 220)
DarkBlue = (0, 0, 100)
Black = (0, 0, 0)
Red = (200, 0, 0)
DarkRed = (100, 0, 0)

# tyhjän lista luominen (arvo, arvo2)
RandomNums = []
MyNums = []
PickedNums = []
NumButtons = []
SpeedButtons = []
GuessedNums = []
YearSelect = []

# tyhjän "sanakirjan" luominen {avain:arvo, avian2:arvo2}
JackpotTimes = {}

NumberList = str([x for x in range(10)])

# ikkunan teko
Window = pygame.display.set_mode((Display_Width, Display_Height))
pygame.display.set_caption('Lotto simulator 2019')

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ '


def GetInput():
    global YearSelect
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        # vuosien valitseminen painettujen nappien perusteella
        if event.type == pygame.KEYDOWN:
            for x in NumberList:
                if x == event.unicode:
                    x = int(x)
                    if not len(YearSelect) == 0 or not x == 0:
                        YearSelect.append(x)

            # vuosien kumaaminen
            if event.key == 8 and len(YearSelect) != 0:
                # listan uudelleen luominen ilman viimeistä lukua
                YearSelect = YearSelect[0:-1]


def GetTotalYears():
    TotalYears = map(str, YearSelect)
    # .join liittää listan monta osaa yhteen
    TotalYears = ''.join(TotalYears)
    return TotalYears


def NumberGen(Nums):
    AllNums = [i for i in range(1, 41)]
    List = []

    for x in range(Nums):
        # valitsee n numeron väliltä 0 - 39 (Len(AllNums) = 40)
        n = (random.randrange(len(AllNums)))
        # lisää listaan AllNums listasta valitun n:ännen numeron (laskenta alkaa 0:sta)
        List.append(AllNums[n])
        # poistaa listasta n:ännen numeron ettei voi valita samaa numeroa uudelleen
        del (AllNums[n])
    return List


def RandomGeneratorActive():
    PickedNums = NumberGen(7)
    return PickedNums


# teksti alustan luominen, mihin itse teksti kirjoitetaan
def text_object(text, font, color):
    textSurface = font.render(text, True, color)
    return textSurface, textSurface.get_rect()


# teksti alustan käyttö sekä piirtäminen näytölle
def message_display(text, x, y, font, color):
    TextSurf, TextRect = text_object(text, font, color)
    TextRect.center = (x, y)
    Window.blit(TextSurf, TextRect)


'------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'


# nappula objectin määritteleminen
class Button(object):
    def __init__(self, msg, x, y, w, h, ic, ac, TextColor, ActiveAction=None, InactiveAction=None, action=None):
        self.msg = msg
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.action = action
        self.ActiveAction = ActiveAction
        self.InactiveAction = InactiveAction
        self.ic = ic
        self.ac = ac
        self.TextColor = TextColor
        self.pressed = False
        self.active = -1

    # nappulan piirtämisen määrittely
    def Draw(self, Window):
        if self.active == -1:
            self.color = self.ic
        elif self.active == 1:
            self.color = self.ac

        pygame.draw.rect(Window, self.color, (self.x, self.y, self.w, self.h))
        message_display(self.msg, (self.x + self.w / 2), (self.y + self.h / 2), smallText, self.TextColor)

    # nappula aktivaati käyttäytymisen määrittely
    def activation(self):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if (self.x + self.w) > mouse[0] > self.x and (self.y + self.h) > mouse[1] > self.y:
            if click[0] == 1:
                if not self.pressed:
                    if self.active == -1:
                        self.active = -self.active
                        self.pressed = True
                        if self.ActiveAction != None:
                            self.ActiveAction()

                    elif self.active == 1:
                        self.pressed = True
                        self.active = -self.active
                        if self.InactiveAction != None:
                            self.InactiveAction()
                else:
                    pass

        if self.pressed:
            if click[0] == 0 or not ((self.x + self.w) > mouse[0] > self.x and (self.y + self.h) > mouse[1] > self.y):
                self.pressed = False
        else:
            pass


'------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'


# erilaisen nappulaobjectin määrittely aikaisempaa nappulaa hyväksikäyttäen
class NumButton(Button):
    def activation(self):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if (self.x + self.w) > mouse[0] > self.x and (self.y + self.h) > mouse[1] > self.y:
            if click[0] == 1:
                if not self.pressed:
                    if PickedNums.count(int(self.msg)) == 0 and len(PickedNums) < 7:
                        self.pressed = True
                        self.active = -self.active
                        PickedNums.append(int(self.msg))

                    elif PickedNums.count(int(self.msg)) == 1:
                        self.pressed = True
                        self.active = -self.active
                        PickedNums.remove(int(self.msg))

        if self.pressed:
            if click[0] == 0 or not ((self.x + self.w) > mouse[0] > self.x and (self.y + self.h) > mouse[1] > self.y):
                self.pressed = False
        else:
            pass


'------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

# nappulaobjectien luominen sekä määrittäminen
RandomGenerator = Button('Random', 25, 280, 145, 100, Blue, DarkBlue, Black, RandomGeneratorActive)
StartButton = Button('Start', 625, 280, 145, 100, Red, DarkRed, Black)
StopButton = Button('Stop', 50, 400, 145, 100, Red, DarkRed, Black)

'------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'


def DrawStart():
    global NumButtons
    Window.fill(Grey)
    TotalYears = GetTotalYears()

    # erilaisten tekstien piirtäminen näytölle
    message_display('Pick 7 numbers or choose Random', Display_Width / 2, 20, smallText, White)
    message_display('Select total years with numberkeys:', Display_Width / 2, 300, smallText, White)
    message_display(TotalYears, Display_Width / 2, 320, smallText, White)

    # nappuloiden piirtäminen näytölle
    for numbutton in NumButtons:
        numbutton.Draw(Window)
    RandomGenerator.Draw(Window)
    StartButton.Draw(Window)

    # näytön itse piirtäminen
    pygame.display.update()


def DrawMain():
    global YearsRun
    TotalYears = GetTotalYears()
    Window.fill(Grey)

    MSG = 'Lottoing speed = {} lotto weeks/s'.format(speed)
    message_display(MSG, Display_Width / 2, 20, smallText, White)

    MSG = 'Years total: {} Years left: {}'.format(int(TotalYears), int(TotalYears) - YearsRun)
    message_display(MSG, Display_Width / 2, 120, smallText, White)

    MSG = 'Total jackpots won: {}'.format(Jackpots)
    message_display(MSG, 550, 160, smallText, White)

    for x in range(len(MyNums)):
        Count = GuessedNums.count(x + 1)
        MSG = 'You got {} matches {} times'.format(str(x + 1), Count, round(Count / (TotalWins + 1) * 100, 8))
        message_display(MSG, 180, 160 + 20 * x, smallText, White)

    for speedbutton in SpeedButtons:
        speedbutton.Draw(Window)

    # reset_button.Draw(Window)

    StopButton.Draw(Window)
    pygame.display.update()


def DrawEndScreen():
    global JackpotTimes

    Window.fill(Grey)
    TotalYears = GetTotalYears()

    MSG = 'In {} years you have gotten:'.format(YearsRun)
    message_display(MSG, Display_Width / 2, 20, bigtext, White)

    for x in range(len(MyNums)):
        Count = GuessedNums.count(x + 1)

        MSG = '{} matches {} times, that is {}% of the total wins'.format(str(x + 1), Count,
                                                                          round(Count / TotalWins * 100, 5))
        message_display(MSG, Display_Width / 2, 60 + 20 * x, smallText, White)

    MSG = 'Total jackpots gotten: {}'.format(Jackpots)
    message_display(MSG, Display_Width / 2, 220, smallText, White)

    if len(JackpotTimes) > 0:
        message_display('Jackpot years and weeks:', Display_Width / 2, 250, smallText, White)

        n = 0
        for x in JackpotTimes:
            MSG = 'Year: {}'.format(x)
            message_display(MSG, 300, 270 + 20 * n, smallText, White)

            MSG = 'Week: {}'.format(JackpotTimes[x])
            message_display(MSG, 500, 270 + 20 * n, smallText, White)

            n += 1

    pygame.display.update()


'------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'


def StartMenu():
    DrawStart()
    # numeronappuloiden X ja Y koordinaatti listan luominen (HUOM for i ja for x)
    NumButtonX = [i for i in range(4) for i in range(10)]
    NumButtonY = [i for i in range(4) for x in range(10)]

    for x in range(40):
        numbutton = NumButton(str((x + 1)), (25 + NumButtonX[x] * 75), (40 + NumButtonY[x] * 55), 70, 50, Green,
                              DarkGreen, Black)
        NumButtons.append(numbutton)

    while True:
        GetInput()
        TotalYears = GetTotalYears()
        RandomGenerator.activation()

        if RandomGenerator.active == -1:
            for numbutton in NumButtons:
                # jokaista nappulaobjectia kohden tekee kyseisen nappulan activaation ja toiminnon
                numbutton.activation()
                if PickedNums.count(int(numbutton.msg)) == 1:
                    numbutton.active = 1

        # random generaattorin ollessa aktiivinen, nappulat menevät pois päältä
        elif RandomGenerator.active == 1:
            for numbutton in NumButtons:
                numbutton.active = -1
            RandomNums = RandomGeneratorActive()

        # activoituu vain jos kaikki kriteerit on täytetty
        if len(PickedNums) == 7 or RandomGenerator.active == 1:
            if len(YearSelect) != 0:
                StartButton.activation()
                if StartButton.active == 1:
                    # riippuen onko random generaattori aktiivinen, valitsee joko valitut numerot tai random numerot arvottavaksi
                    if RandomGenerator.active == 1:
                        List = RandomNums
                    else:
                        List = PickedNums
                    return List

        DrawStart()


def Main():
    global speed, YearsRun, WeeksRun, BiggestWin, TotalWins, Jackpots, MyNums
    DrawMain()
    TotalYears = GetTotalYears()
    TotalYears = int(TotalYears)
    MyNums = StartMenu()

    SpeedbuttonNum = [i for i in range(5)]
    SpeedbuttonSpeed = [10 ** (1 + i) for i in range(5)]
    SpeedButtonMSG = []

    for x in range(len(SpeedbuttonSpeed)):
        msg = ''.join('{}/s'.format(SpeedbuttonSpeed[x]))
        SpeedButtonMSG.append(msg)

    for x in range(5):
        speedbutton = Button(str(SpeedButtonMSG[x]), 150 + SpeedbuttonNum[x] * 100, 40, 90, 60, LightGrey, DarkGrey,
                             White)
        SpeedButtons.append(speedbutton)

    reset_button = Button(str("reset"), 600 + 600, 40, 90, 60, LightGrey, DarkGrey, White)

    while YearsRun < TotalYears:
        GetInput()
        Winners = 0
        WinnerNums = NumberGen(8)

        for speedbutton in SpeedButtons:
            speedbutton.activation()
            StopButton.activation()
            if speedbutton.active == 1:
                speed = SpeedButtonMSG.index(speedbutton.msg)
                speed = SpeedbuttonSpeed[speed]
                if speedbutton.pressed == False:
                    speedbutton.active = -1

        reset_button.activation()
        if reset_button.active == 1:

            if speedbutton.pressed == False:
                speedbutton.active = -1

        for num in MyNums:
            for win in WinnerNums:
                # vertaa omaa numeroa kaikkiin voittonumeroihin
                if num == win:
                    Winners += 1

        if Winners > 0:
            # lisää voittaneiden numeroiden määrän listaan
            GuessedNums.append(Winners)

        if Winners == len(MyNums):
            Jackpots += 1
            # lisää listaan jackpot vuoden ja viikon (avain:arvo) parina
            JackpotTimes.update({YearsRun: WeeksRun})

        WeeksRun += 1
        if WeeksRun == 52:
            WeeksRun = 0
            YearsRun += 1

        TotalWins += Winners
        WinPercentage = (TotalWins / (52 * YearsRun + WeeksRun))
        WinPercentage = round(WinPercentage, 5)

        if speed == 10 and (WeeksRun / 2).is_integer():
            DrawMain()
        elif speed == 100 and (WeeksRun / 5).is_integer():
            DrawMain()
        elif speed == 1000 and WeeksRun == 0:
            DrawMain()
        elif (400 * YearsRun / speed).is_integer() and WeeksRun == 0:
            DrawMain()

        if StopButton.active == 1:
            break

        if speed < 100000:
            clock.tick(speed)


def EndScreen():
    while True:
        DrawEndScreen()
        GetInput()


'------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

StartMenu()
Main()
EndScreen()
