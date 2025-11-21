import qualified Data.Map as Map
import Text.Read
import Data.List


data InputCommand = Add String String String String 
                  | Find String 
                  | Quit 
                  | Err

strToCommand :: String -> InputCommand
strToCommand "quit" = Quit
strToCommand input = 
    case words input of
        ["find", name] -> Find name
        ["add", name, phone_type, country_code, phone_no] -> Add name phone_type country_code phone_no
        _ -> Err



type Name = String
data PhoneBook = Empty | Node String [Phone] PhoneBook PhoneBook
    deriving (Show,Eq)

data PhoneType = WorkLandline | PrivateMobile | WorkMobile | Other
    deriving (Show, Eq, Read)

data CountryCode = CountryCode Integer
    deriving (Eq)

instance Show CountryCode where
    show (CountryCode code) = show code

data PhoneNo = PhoneNo Integer
    deriving (Eq)

instance Show PhoneNo where
    show (PhoneNo num) = show num


data Phone = Phone {
    phoneType :: Maybe PhoneType,
    countryCode :: Maybe CountryCode,
    phoneNo :: PhoneNo
}

instance Eq Phone where
    (Phone _ _ num1) == (Phone _ _ num2) = num1 == num2



instance Show Phone where
    show Phone {phoneType = (Just phone_type), countryCode = (Just country), phoneNo = num} =
        "+" ++ show country ++ " " ++ show num ++ " (" ++ show phone_type ++ ")"

    show Phone {phoneType = Nothing, countryCode = (Just country), phoneNo = num} =
        "+" ++ show country ++ " " ++ show num

    show Phone {phoneType = (Just phone_type), countryCode = Nothing, phoneNo = num} =
        show num ++ " (" ++ show phone_type ++ ")"

    show Phone {phoneType = Nothing, countryCode = Nothing, phoneNo = num} =
        show num


toCountryCode :: Integer -> CountryCode
toCountryCode num
    | num < 0 = error "Negative country code"
    | otherwise = CountryCode num

toPhoneNo :: Integer -> PhoneNo
toPhoneNo num
    | num < 0 = error "Negative phone number"
    | otherwise = PhoneNo num


readPhoneType :: String -> Maybe PhoneType
readPhoneType "" = Nothing
readPhoneType input_str =
    case readMaybe input_str of
        Just value -> Just value
        Nothing -> error "Incorrect phone type"


readCountryCode :: String -> Maybe CountryCode
readCountryCode "" = Nothing
readCountryCode input_str =
    country_code where
        number :: Integer
        number = case readMaybe input_str of
            Just value -> value
            _             -> error "Incorrect country code"

        country_code =
            if number >= 0
            then Just (toCountryCode number)
            else error "Negative country code"

readPhoneNo :: String -> PhoneNo
readPhoneNo "" = error "Incorrect phone number"
readPhoneNo input_str =
    phone_num where
        number :: Integer
        number = case readMaybe input_str of
            Just value -> value
            _          -> error "Incorrect phone number"

        phone_num =
            if number >= 0
            then toPhoneNo number
            else error "Negative phone number"


readPhone :: String -> String -> String -> Phone
readPhone phonetypestr countrycodestr phonenostr =
    Phone {phoneType = readPhoneType phonetypestr, countryCode = readCountryCode countrycodestr, phoneNo = readPhoneNo phonenostr}


findEntries :: Name -> PhoneBook -> [Phone]
findEntries name Empty = []
findEntries name (Node node_name node_phones left_book right_book)
  | name == node_name = node_phones
  | name < node_name = findEntries name left_book
  | name > node_name = findEntries name right_book

addEntry :: Name -> String -> String -> String -> PhoneBook -> PhoneBook
addEntry name phonetype ccode phonenum currentbook =
    phonebook where
        new_phone = readPhone phonetype ccode phonenum

        recursive_add_entry :: Name -> Phone -> PhoneBook -> PhoneBook
        recursive_add_entry name phone Empty = Node name [phone] Empty Empty
        recursive_add_entry name phone (Node node_name node_phones left_book right_book)
            | name == node_name =
                if phone `elem` node_phones
                    then Node node_name node_phones left_book right_book
                else
                    Node node_name (phone:node_phones) left_book right_book

            | name < node_name = Node node_name node_phones (recursive_add_entry name phone left_book) right_book
            | name > node_name = Node node_name node_phones left_book (recursive_add_entry name phone right_book)

        phonebook = recursive_add_entry name new_phone currentbook

emptyBook :: PhoneBook
emptyBook = Empty


eventLoop :: PhoneBook -> IO ()
eventLoop current_book = do
        line <- getLine
        putStrLn $ "> " ++ line
        let command = strToCommand line
        case command of
            Quit -> return ()
            Err -> do
                putStrLn "I cannot do that"
                eventLoop current_book
            (Find name) -> do
                case findEntries name current_book of
                    [] -> putStrLn "Not found"
                    phones -> do 
                        let phone_strs = sort [show phone | phone <- phones]
                        putStrLn (intercalate ", " phone_strs)

                eventLoop current_book
            (Add name phone_type country_code phone_no) -> do
                let new_book = addEntry name phone_type country_code phone_no current_book
                putStrLn "Done"
                eventLoop new_book

main = do
    putStrLn "Welcome to phone book application"
    () <- eventLoop emptyBook
    putStrLn "Bye"
