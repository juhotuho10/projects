import Control.Applicative ((<|>))
import Data.List (intercalate, sortBy)
import Data.Time (Day, defaultTimeLocale, parseTimeM)
import Data.Time.Calendar (addDays)
import Distribution.Fields (SectionArg (SecArgName))
import Text.Parsec (ParseError, char, eof, many1, noneOf, parse, string, try)
import Text.Parsec.String (Parser)

parseDate :: String -> Maybe Day
parseDate = parseTimeM True defaultTimeLocale "%Y-%m-%d"

type EventName = String
type EventPlace = String
type EventDate = Day

data InputErr = BadCommand | BadDate
    deriving (Show, Eq)

data Search
    = EventByName EventName
    | EventByPlace EventPlace
    | EventByDate EventDate
    deriving (Show, Eq)

data InputCommand
    = AddEvent Event
    | GetEvent Search
    | QuitCmd
    | CmdErr InputErr

data Event = Event {event_name :: EventName, event_place :: EventPlace, event_time :: EventDate}
    deriving (Show, Eq)

data OutputEvent = Err InputErr | Quit | Printout String
    deriving (Show, Eq)

data Query
    = AddEventQuery String String String
    | EventByNameQuery String
    | EventByPlaceQuery String
    | EventByDateQuery String
    | QuitQuery
    deriving (Show)

bracketParser :: Parser String
bracketParser = do
    char '['
    content <- many1 (noneOf "]")
    char ']'
    return $ trim content
  where
    trim = unwords . words

queryParser :: Parser Query
queryParser = quitParser <|> addEventParser <|> eventByNameParser <|> eventByPlaceParser <|> eventByDateParser
  where
    quitParser = do
        try $ string "Quit"
        eof
        return QuitQuery

    addEventParser = do
        try $ string "Event "
        name <- bracketParser
        try $ string " happens at "
        place <- bracketParser
        try $ string " on "
        date <- many1 (noneOf "\n")
        return $ AddEventQuery name place (unwords . words $ date)

    eventByNameParser = do
        try $ string "Tell me about "
        EventByNameQuery <$> bracketParser

    eventByPlaceParser = do
        try $ string "What happens at "
        EventByPlaceQuery <$> bracketParser

    eventByDateParser = do
        try $ string "What happens near "
        date <- many1 (noneOf "\n")
        return $ EventByDateQuery (unwords . words $ date)

parseInput :: String -> Either ParseError Query
parseInput = parse queryParser ""

strToCommand :: String -> InputCommand
strToCommand input =
    command
  where
    parsed = parseInput input
    command = case parsed of
        Right QuitQuery -> QuitCmd
        Right (AddEventQuery name place date_str) ->
            case parseDate date_str of
                Nothing -> CmdErr BadDate
                Just date -> AddEvent Event{event_name = name, event_place = place, event_time = date}
        Right (EventByNameQuery name) ->
            GetEvent (EventByName name)
        Right (EventByDateQuery date_str) ->
            case parseDate date_str of
                Nothing -> CmdErr BadDate
                Just date -> GetEvent (EventByDate date)
        Right (EventByPlaceQuery place) ->
            GetEvent (EventByPlace place)
        Left err -> CmdErr BadCommand

eventOrdering :: Search -> Event -> Event -> Ordering
eventOrdering (EventByDate _) e1 e2
    | event_time e1 == event_time e2 = compare (event_name e1) (event_name e2)
    | otherwise = compare (event_time e1) (event_time e2)
eventOrdering (EventByName _) e1 e2 = compare (event_name e1) (event_name e2)
eventOrdering (EventByPlace _) e1 e2 = compare (event_name e1) (event_name e2)

findCommandsToStr :: Search -> [Event] -> String
findCommandsToStr (EventByName _) [] = "I do not know of such event"
findCommandsToStr (EventByPlace _) [] = "Nothing that I know of"
findCommandsToStr (EventByDate _) [] = "Nothing that I know of"
findCommandsToStr (EventByName _) events = event_strings
  where
    strings = ["Event [ " ++ event_name event ++ " ] happens at [ " ++ event_place event ++ " ] on " ++ show (event_time event) | event <- events]
    event_strings = intercalate "\n" strings
findCommandsToStr (EventByPlace _) events = event_strings
  where
    strings = ["Event [ " ++ event_name event ++ " ] happens at [ " ++ event_place event ++ " ]" | event <- events]
    event_strings = intercalate "\n" strings
findCommandsToStr (EventByDate _) events = event_strings
  where
    strings = ["Event [ " ++ event_name event ++ " ] happens on " ++ show (event_time event) | event <- events]
    event_strings = intercalate "\n" strings

parseCommand :: [Event] -> InputCommand -> ([Event], OutputEvent)
parseCommand current_calendar QuitCmd = (current_calendar, Quit)
parseCommand current_calendar (CmdErr err) = (current_calendar, Err err)
parseCommand current_calendar (AddEvent event) =
    if event `elem` current_calendar
        then (current_calendar, Printout "Event already exists")
        else (event : current_calendar, Printout "Ok")
parseCommand current_calendar (GetEvent eventBy) =
    case eventBy of
        (EventByName name) ->
            (current_calendar, Printout str)
          where
            suitable_events = filter (\x -> event_name x == name) current_calendar
            sorted_events = sortBy (eventOrdering eventBy) suitable_events

            str = findCommandsToStr (EventByName name) sorted_events
        (EventByPlace place) ->
            (current_calendar, Printout str)
          where
            suitable_events = filter (\x -> event_place x == place) current_calendar
            sorted_events = sortBy (eventOrdering eventBy) suitable_events

            str = findCommandsToStr (EventByPlace place) sorted_events
        (EventByDate date) ->
            (current_calendar, Printout str)
          where
            start_date = addDays (-7) date
            end_date = addDays 7 date

            suitable_events = filter (\x -> event_time x >= start_date && event_time x <= end_date) current_calendar
            sorted_events = sortBy (eventOrdering eventBy) suitable_events

            str = findCommandsToStr (EventByDate date) sorted_events

eventLoop :: [Event] -> IO ()
eventLoop current_calendar = do
    line <- getLine
    putStrLn $ "> " ++ line
    let command = strToCommand line
    let (new_calendar, output_event) = parseCommand current_calendar command
    case output_event of
        Printout str -> do
            putStrLn str
            eventLoop new_calendar
        Quit -> return ()
        Err err -> do
            case err of
                BadCommand ->
                    putStrLn "I do not understand that"
                BadDate ->
                    putStrLn "Bad date"

            eventLoop current_calendar

emptyCalendar :: [Event]
emptyCalendar = []

main = do
    () <- eventLoop emptyCalendar
    putStrLn "Bye"
