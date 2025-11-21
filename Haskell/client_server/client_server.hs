{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

{-# HLINT ignore "Use newtype instead of data" #-}

import Control.Concurrent.Chan
import Control.Concurrent.QSemN
import Data.List (intercalate, nub, sort, sortOn)

import Control.Exception
import System.Timeout

import GHC.IO.Encoding (BufferCodec (recover))
import GHC.RTS.Flags (ProfFlags (retainerSelector))

import Control.Concurrent
import System.Random (randomRIO)

import Control.Arrow (Arrow (first))
import GHC.IOPort (newEmptyIOPort)
import GHC.Integer (smallInteger)
import System.IO

type ClientID = Int
type ServerID = Int
type Key = Int
type Value = String

data ServerValues = Values {value :: Value, subscribers :: [ClientID]}
  deriving (Eq)
instance Show ServerValues where
  show (Values v subs) =
    "Values { value = " ++ show v ++ ", subscribers = " ++ show subs ++ " }"

newServerValues :: Value -> [ClientID] -> ServerValues
newServerValues server_value subscribers =
  Values{value = server_value, subscribers = subscribers}

data ClientData = ClientData {client_id :: ClientID}

data ServerState
  = Failed ServerID [(Key, ServerValues)]
  | Online ServerID [(Key, ServerValues)]
  deriving (Eq, Show)

data MessageQueueMessage
  = Client [ClientID] ToClient
  | Server [ServerID] ToServer
data ToClient
  = SendValue (Key, Value)
  | ClientIdle
data ToServer
  = FromClient ClientMessage
  | FromServer ServerMessage
  | ServerIdle

data ClientMessage
  = Subscribe Key ClientID
  | Write Key Value
  | Read Key ClientID
data ServerMessage
  = WriteCopy Key Value
  | Fail
  | Recover
  | GetData ServerID ServerID
  | WriteData [(Key, ServerValues)]

newServeState :: ServerID -> ServerState
newServeState id = Online id []

newClientState :: ClientID -> ClientData
newClientState id = ClientData{client_id = id}

increment :: QSemN -> IO ()
increment semaphore = signalQSemN semaphore 1

updateSharedState :: ServerID -> ServerState -> MVar [(ServerID, ServerState)] -> IO ()
-- updates teh server ID to a new ServerState in MVar
updateSharedState id new_state stateMVar =
  modifyMVar_ stateMVar $ \server_status_list -> do
    let filtered_states = filter (\(k, _) -> k /= id) server_status_list
    let updated_states = (id, new_state) : filtered_states
    return updated_states

runClient :: ClientData -> MVar ToClient -> QSemN -> IO ()
-- receive client messages and execute them
runClient client_data message_queue tasks_done = do
  -- print "increment"
  increment tasks_done
  message <- takeMVar message_queue -- blocks until message is received
  -- print "start client loop"
  case message of
    SendValue server_data -> do
      print "send value received:"
      print server_data
    ClientIdle -> return () -- do nothing
    -- print "end client loop"
  runClient client_data message_queue tasks_done

getOtherServerIDs :: ServerID -> MVar [(ServerID, ServerState)] -> IO [Int]
-- get all server IDs except own ID
getOtherServerIDs own_id shared_server_states_mvar = do
  servers <- readMVar shared_server_states_mvar
  let all_ids = map fst servers
  return [id | id <- all_ids, id /= own_id]

updateList :: [(Int, a)] -> (Int, a) -> [(Int, a)]
updateList key_data (key, new_value) = new_list
 where
  deleted_key = [d | d <- key_data, fst d /= key]
  new_list = (key, new_value) : deleted_key

getKeyValue :: [(Key, a)] -> Key -> a
getKeyValue values key =
  case lookup key values of
    Just keyData -> keyData
    Nothing -> do
      error ("key " ++ show key ++ " not found in the data")

notifyAllSubs :: Key -> ServerValues -> Chan MessageQueueMessage -> IO ()
-- writes a new task to return tasks to notify all the subscribers of a key
notifyAllSubs key server_values return_channel = do
  let subs = subscribers server_values
  let key_value = value server_values
  writeChan return_channel (Client subs (SendValue (key, key_value)))

notifySubs :: Key -> Value -> [ClientID] -> Chan MessageQueueMessage -> IO ()
-- notify only specific clients for a key value
notifySubs key key_value subs return_channel = do
  writeChan return_channel (Client subs (SendValue (key, key_value)))

addSubscriber :: [(Key, ServerValues)] -> Key -> ClientID -> [(Key, ServerValues)]
addSubscriber values key id = newValues
 where
  keyData = getKeyValue values key
  oldSubs = subscribers keyData
  newSubs = nub (id : oldSubs)
  newKeyData = keyData{subscribers = newSubs}
  newValues = updateList values (key, newKeyData)

writeValue :: [(Key, ServerValues)] -> Key -> Value -> IO [(Key, ServerValues)]
-- add new value to server values for a key
writeValue old_values key value = do
  case lookup key old_values of
    Just keyData -> do
      new_values <- case lookup key old_values of
        Just old_values -> return old_values{value = value}
        Nothing -> return Values{value = value, subscribers = []}
      let subs_to_notify = subscribers new_values
      let new_list = updateList old_values (key, new_values)
      return new_list
    Nothing -> do
      return ((key, newServerValues value []) : old_values)

prettyServerStates :: [(ServerID, ServerState)] -> IO ()
-- print function for debugging
-- alias so you can call println like in other languages
prettyServerStates xs = do
  case xs of
    [] -> putStrLn "∅"
    _ -> mapM_ putStrLn (concatMap ppLines xs)
 where
  ppLines :: (ServerID, ServerState) -> [String]
  ppLines (sid, Failed _ kvs) =
    ("Server " ++ show sid ++ ": FAILED")
      : if null kvs
        then ["(no keys)"]
        else [show k ++ " -> " ++ show v | (k, v) <- kvs]
  ppLines (sid, Online _ kvs) =
    ("Server " ++ show sid ++ ": ONLINE")
      : if null kvs
        then ["(no keys)"]
        else [show k ++ " -> " ++ show v | (k, v) <- kvs]

getServerKeyValues :: ServerState -> [(Key, ServerValues)]
getServerKeyValues serverState =
  case serverState of
    Online _ kvs ->
      kvs
    Failed _ kvs ->
      kvs

runServer :: MVar [(ServerID, ServerState)] -> ServerID -> ServerState -> MVar ToServer -> QSemN -> Chan MessageQueueMessage -> IO ()
-- waits for messages for server and handle them
runServer shared_server_states_mvar current_id current_state message_queue tasks_done return_channel = do
  increment tasks_done

  -- print "increment"
  servers <- readMVar shared_server_states_mvar
  -- putStrLn ""
  -- case current_id of
  --  0 -> prettyServerStates servers
  --  id -> return ()
  -- putStrLn ""

  -- print "waiting fro message"
  message <- takeMVar message_queue -- blocks until message is received
  -- print "start server loop"
  -- print ("new message: " ++ toServerMessageName message)
  new_state <- case current_state of
    Failed id _ ->
      case message of
        FromServer Recover -> do
          -- pick random ID from other IDs to request recovery data from
          other_ids <- getOtherServerIDs current_id shared_server_states_mvar
          let len = length other_ids
          if len == 0
            then
              error "cannot recover from failure, the server has no other servers to fetch the lost data from"
            else do
              index <- randomRIO (0, len - 1)
              let random_id = other_ids !! index

              -- request data from own id to be sent to own id
              writeChan return_channel (Server [random_id] (FromServer (GetData id id)))

          return (newServeState id) -- recover to no state server, get previous data back later
        _ -> do
          return current_state -- do nothing, server still in Failed state
    Online id server_data ->
      case message of
        FromClient clientmessage ->
          case clientmessage of
            Subscribe key client_id -> do
              let new_state = Online id (addSubscriber server_data key client_id)
              updateSharedState id new_state shared_server_states_mvar
              return new_state
            Write key value -> do
              -- send WriteCopy to everyone, including self
              servers <- readMVar shared_server_states_mvar
              let all_ids = map fst servers
              writeChan return_channel (Server all_ids (FromServer (WriteCopy key value)))
              return current_state -- no change right now, it will be written with WriteCopy
            Read key client_id -> do
              -- send key value to client_id
              let key_values = getKeyValue server_data key
              let key_value = value key_values
              notifySubs key key_value [client_id] return_channel

              return current_state -- no state change
        FromServer server_message ->
          case server_message of
            WriteCopy key value -> do
              -- write new data
              new_data <- writeValue server_data key value
              -- notify subscribers
              let key_values = getKeyValue new_data key
              let subs = subscribers key_values
              notifySubs key value subs return_channel

              -- update shared state
              let new_state = Online id new_data
              updateSharedState id new_state shared_server_states_mvar
              return new_state -- new state
            Fail -> do
              -- fail and stop responding to messages until recovered
              let new_state = Failed id server_data
              updateSharedState id new_state shared_server_states_mvar
              return new_state
            Recover -> do
              return current_state -- does nothing, server already up
            GetData data_from_server_id data_to_server_id -> do
              servers <- readMVar shared_server_states_mvar

              from_server_data <- case lookup data_from_server_id servers of
                Just keyData -> do
                  return keyData
                Nothing -> error "key not found in the data"
              let keyValues = getServerKeyValues from_server_data
              writeChan return_channel (Server [data_to_server_id] (FromServer (WriteData keyValues)))

              return current_state -- no change
            WriteData key_values -> do
              let new_state = Online id key_values
              updateSharedState id new_state shared_server_states_mvar
              return new_state -- replace state with new one
        ServerIdle -> do
          return current_state -- do nothin
          -- print "end server loop"
  runServer shared_server_states_mvar current_id new_state message_queue tasks_done return_channel

initServers :: ServerID -> MVar [(ServerID, ServerState)] -> QSemN -> [MVar ToServer] -> Chan MessageQueueMessage -> IO ()
-- create servers with inital state
initServers _ _ _ [] _ = return ()
initServers id server_states_mvar tasks_done (queue : rest) return_tasks = do
  _ <- forkIO (runServer server_states_mvar id (newServeState id) queue tasks_done return_tasks)
  initServers (id + 1) server_states_mvar tasks_done rest return_tasks

initClients :: ClientID -> QSemN -> [MVar ToClient] -> IO ()
-- create clients with inital state
initClients _ _ [] = return ()
initClients id tasks_done (queue : rest) = do
  _ <- forkIO (runClient (newClientState id) queue tasks_done)
  initClients (id + 1) tasks_done rest

addServerJobs :: ServerID -> [ServerID] -> ToServer -> [MVar ToServer] -> IO ()
-- distribute jobs to the servers, each job has server IDs that are attached to them
addServerJobs _ [] _ [] = return ()
addServerJobs _ nums_left _ [] = error "a job were givent to ids that don't exist"
-- add idle jobs to the rest since no more jobs available

addServerJobs current_num [] job_type (nextMVar : restMVars) = do
  available <- tryPutMVar nextMVar ServerIdle
  if available
    then do
      addServerJobs (current_num + 1) [] job_type restMVars
    else do
      error "the communication channel should never be full here"

-- distribute the jobs to the receivers, or if they dont have a job num allocated, let them idle
addServerJobs current_num job_server_nums@(next_num : restNums) job_type (nextMVar : restMVars) = do
  if current_num == next_num
    then do
      available <- tryPutMVar nextMVar job_type
      if available
        then do
          addServerJobs (current_num + 1) restNums job_type restMVars
        else do
          error "the communication channel should never be full here"
    else do
      available <- tryPutMVar nextMVar ServerIdle
      if available
        then do
          addServerJobs (current_num + 1) job_server_nums job_type restMVars
        else do
          error "the communication channel should never be full here"

addClientJobs :: ClientID -> [ClientID] -> ToClient -> [MVar ToClient] -> IO ()
-- distribute jobs to the clients, each job has clients IDs that are attached to them
addClientJobs _ [] _ [] = return ()
addClientJobs _ nums_left _ [] = error "some jobs were givent to ids that don't exist"
-- add idle jobs to the rest since no more jobs available
addClientJobs current_num [] job_type (nextMVar : restMVars) = do
  available <- tryPutMVar nextMVar ClientIdle
  if available
    then do
      addClientJobs (current_num + 1) [] job_type restMVars
    else do
      error "the communication channel should never be full here"

-- distribute the jobs to the receivers, or if they dont have a job num allocated, let them idle
addClientJobs current_num job_client_nums@(next_num : restNums) job_type (nextMVar : restMVars) = do
  if current_num == next_num
    then do
      available <- tryPutMVar nextMVar job_type
      if available
        then do
          addClientJobs (current_num + 1) restNums job_type restMVars
        else do
          error "the communication channel should never be full here"
    else do
      available <- tryPutMVar nextMVar ClientIdle
      if available
        then do
          addClientJobs (current_num + 1) job_client_nums job_type restMVars
        else do
          error "the communication channel should never be full here"

messageNames :: [MessageQueueMessage] -> [String]
-- for debug printing message queue
messageNames = map getName
 where
  getName :: MessageQueueMessage -> String
  getName (Client _ toClient) = toClientMessageName toClient
  getName (Server _ toServer) = toServerMessageName toServer

toClientMessageName :: ToClient -> String
-- for debug printing ToClient messages
toClientMessageName sm =
  case sm of
    SendValue _ -> "SendValue"
    ClientIdle -> "ClientIdle"

toServerMessageName :: ToServer -> String
-- for debug printing ToServer messages
toServerMessageName sm =
  case sm of
    FromClient cm -> case cm of
      Subscribe _ _ -> "Subscribe"
      Write _ _ -> "Write"
      Read _ _ -> "Read"
    FromServer sm -> case sm of
      WriteCopy _ _ -> "WriteCopy"
      Fail -> "Fail"
      Recover -> "Recover"
      GetData _ _ -> "GetData"
      WriteData _ -> "WriteData"
    ServerIdle -> "ServerIdle"

readAvailableMessages :: Chan a -> IO [a]
-- helper for clearing the channel non lazily
readAvailableMessages chan = go []
 where
  tryReadChanTimeout :: Chan a -> Int -> IO (Maybe a)
  tryReadChanTimeout chan micros = timeout micros (readChan chan)

  go acc = do
    result <- tryReadChanTimeout chan 1000
    case result of
      Just msg -> go (msg : acc)
      Nothing -> return (reverse acc)

runTasks :: [MessageQueueMessage] -> Chan MessageQueueMessage -> [MVar ToServer] -> [MVar ToClient] -> QSemN -> QSemN -> Int -> Int -> IO ()
runTasks [] _ _ _ _ _ _ _ = return ()
runTasks (message : rest) return_tasks server_tasks client_tasks servers_done clients_done n_servers n_clients = do
  -- match message and send message to receivers
  -- sends 1 message at a time to all the receivers, waits for everyone to complete task in parallel
  -- once everyone has completed the task it sends the next task
  case message of
    Client ids client_message -> do
      -- running client tasks
      let sorted_ids = sort ids -- list of client ids that the job is pointed to
      addClientJobs 0 sorted_ids client_message client_tasks
      waitQSemN clients_done n_clients -- wait until everyone is ready
    Server ids server_message -> do
      -- running server tasks
      let sorted_ids = sort ids -- list of server ids that the job is pointed to
      addServerJobs 0 sorted_ids server_message server_tasks
      waitQSemN servers_done n_servers -- wait until everyone is ready

  -- read what additional things we need to add to the queue
  new_tasks <- readAvailableMessages return_tasks
  let upcoming_tasks = new_tasks ++ rest
  -- if not (null upcoming_tasks)
  --  then do
  --    --print "upcoming tasks:"
  --    --print (messageNames upcoming_tasks)
  --    --print "end task"
  --    --print ""
  --  else do
  --    --print ()

  -- go next iteration to process next command
  runTasks upcoming_tasks return_tasks server_tasks client_tasks servers_done clients_done n_servers n_clients

setupServers :: [MessageQueueMessage] -> Int -> Int -> IO [(ServerID, ServerState)]
setupServers messageQueue numServers numClients = do
  serverTasksDone <- newQSemN 0
  clientTasksDone <- newQSemN 0
  returnTasks <- newChan

  -- create base server state for all servers
  let new_server_states = [(id, newServeState id) | id <- [0 .. (numServers - 1)]]

  -- shared state for all servers
  server_states_mvar <- newEmptyMVar
  putMVar server_states_mvar new_server_states

  -- creates a taskqueue for all the servers
  server_tasks <- mapM (const newEmptyMVar) [1 .. numServers]
  -- starts running all the server, have them wait for tasks
  initServers 0 server_states_mvar serverTasksDone server_tasks returnTasks

  -- creates a taskqueue for all the clients
  client_tasks <- mapM (const newEmptyMVar) [1 .. numClients]
  -- starts running all the clients, have them wait for tasks
  initClients 0 clientTasksDone client_tasks

  -- waits for all clients and servers to have started
  waitQSemN serverTasksDone numServers
  waitQSemN clientTasksDone numClients

  -- stars running thought tasks in the tasks queue
  runTasks messageQueue returnTasks server_tasks client_tasks serverTasksDone clientTasksDone numServers numClients

  final_state <- takeMVar server_states_mvar

  return (sortOn fst final_state)

failAssert :: Bool -> Int -> IO ()
failAssert True testNum = do
  print ("Test " ++ show testNum ++ " Success")
  return ()
failAssert False testNum = do
  error ("Test " ++ show testNum ++ " Assertion failed: doesn't match wanted state")

runTests :: IO ()
runTests = do
  -- ============================================================================
  -- test for write working
  let numServers = 2
  let numClients = 1
  let messages = [Server [0] (FromClient (Write 0 "hello"))]
  final_state <- setupServers messages numServers numClients

  let wanted_state =
        [ (0, Online 0 [(0, newServerValues "hello" [])])
        , (1, Online 1 [(0, newServerValues "hello" [])])
        ]

  failAssert (wanted_state == final_state) 1

  -- ============================================================================
  -- test for server failure and recovery working
  let numServers = 2
  let numClients = 1
  let messages =
        [ Server [0] (FromServer Fail)
        , Server [1] (FromClient (Write 0 "hello"))
        , Server [0] (FromServer Recover)
        ]
  final_state <- setupServers messages numServers numClients

  let wanted_state =
        [ (0, Online 0 [])
        , (1, Online 1 [(0, newServerValues "hello" [])])
        ]

  failAssert (wanted_state == final_state) 2

  -- ============================================================================
  -- test for subscribing to values working
  let numServers = 2
  let numClients = 1
  let messages =
        [ Server [1] (FromClient (Write 0 "hello"))
        , Server [1] (FromClient (Subscribe 0 0))
        ]
  final_state <- setupServers messages numServers numClients

  let wanted_state =
        [ (0, Online 0 [(0, newServerValues "hello" [])])
        , (1, Online 1 [(0, newServerValues "hello" [0])])
        ]

  failAssert (wanted_state == final_state) 3

  -- ============================================================================
  -- test for read working, should print out (0,"world") twice, once when subscriber is notified and once after read
  let numServers = 2
  let numClients = 1
  let messages =
        [ Server [1] (FromServer (WriteCopy 0 "hello"))
        , Server [1] (FromClient (Subscribe 0 0))
        , Server [1] (FromServer (WriteCopy 0 "world"))
        , Server [1] (FromClient (Read 0 0))
        ]
  final_state <- setupServers messages numServers numClients

  let wanted_state =
        [ (0, Online 0 [])
        , (1, Online 1 [(0, newServerValues "world" [0])])
        ]

  failAssert (wanted_state == final_state) 4

  -- ============================================================================

  -- test for get data working
  let numServers = 2
  let numClients = 1
  let messages =
        [ Server [1] (FromServer (WriteCopy 0 "hello"))
        , Server [1] (FromServer (GetData 1 0))
        ]
  final_state <- setupServers messages numServers numClients

  let wanted_state =
        [ (0, Online 0 [(0, newServerValues "hello" [])])
        , (1, Online 1 [(0, newServerValues "hello" [])])
        ]

  failAssert (wanted_state == final_state) 5

  -- ============================================================================

  -- test for write data
  let numServers = 2
  let numClients = 1
  let messages =
        [ Server [0] (FromServer (WriteData [(0, newServerValues "hello" [])]))
        ]
  final_state <- setupServers messages numServers numClients

  let wanted_state =
        [ (0, Online 0 [(0, newServerValues "hello" [])])
        , (1, Online 1 [])
        ]

  failAssert (wanted_state == final_state) 6

  -- ============================================================================

  -- test for read
  let numServers = 2
  let numClients = 1
  let messages =
        [ Server [0] (FromServer (WriteData [(0, newServerValues "hello" [])]))
        ]
  final_state <- setupServers messages numServers numClients

  let wanted_state =
        [ (0, Online 0 [(0, newServerValues "hello" [])])
        , (1, Online 1 [])
        ]

  failAssert (wanted_state == final_state) 7

-- ============================================================================

main :: IO ()
main = do
  runTests
  putStrLn "end"
