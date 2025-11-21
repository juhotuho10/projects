import qualified Data.Map
import Text.Read

data InputCommand = Encode Integer [String] | Decode Integer [String] | Quit | Err

strToCommand :: String -> InputCommand
strToCommand "" = Err
strToCommand "quit" = Quit
strToCommand input | length (words input) < 3 = Err
strToCommand input = 
    command where
        (input_command:maybeNum:rest) = words input
        command = case (input_command, readMaybe maybeNum) of
            ("encode", Just num) -> Encode num rest
            ("decode", Just num) -> Decode num rest
            (_ , _) -> Err



encode :: Int -> String -> String
encode shift = map (charmap Data.Map.!)
    where
        charlist = ['0'..'9'] ++ ['A'..'Z'] ++ ['a'..'z']
        listlength = length charlist
        shiftedlist = take listlength (drop (shift `mod` listlength) (cycle charlist))
        charmap = Data.Map.fromList $ zip charlist shiftedlist

decode :: Int -> String -> String
decode shift = encode (negate shift)

eventLoop :: IO String
eventLoop = do
        line <- getLine
        putStrLn $ "> " ++ line
        let command = strToCommand line
        case command of
            Quit -> return line
            Err -> do
                putStrLn "I cannot do that"
                eventLoop
            (Encode num strings) -> do
                let encoder = encode (fromIntegral num)
                let encoded_str = unwords $ map encoder strings
                putStrLn encoded_str
                eventLoop
            (Decode num strings) -> do
                let decoder = decode (fromIntegral num)
                let decoded_str = unwords $ map decoder strings
                putStrLn decoded_str
                eventLoop

main = do
    final_state <- eventLoop
    putStrLn "Bye"
