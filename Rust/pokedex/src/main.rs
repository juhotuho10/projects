#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

use iced::{
    Bottom, Center, Element, Fill, Left, Task, Theme, futures,
    widget::{
        button, center, column, container,
        image::{Handle, Image},
        row, text,
    },
};

use bytes::Bytes;
use reqwest::Response;

use std::{
    collections::HashMap,
    io,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use iced_gif::{Frames, Gif};
use rodio::{Decoder, DeviceSinkBuilder, MixerDeviceSink, Player};

static TYPE_IMAGE_CACHE: OnceLock<Mutex<HashMap<String, Handle>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<String, Handle>> {
    TYPE_IMAGE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

static CLIENT_CACHE: OnceLock<reqwest::Client> = OnceLock::new();

fn client_fetch(url: &str) -> impl Future<Output = Result<Response, reqwest::Error>> {
    CLIENT_CACHE
        .get_or_init(reqwest::Client::new)
        .get(url)
        .send()
}

static RUNTIM_CACHE: OnceLock<Mutex<Counter>> = OnceLock::new();

fn runtime_cache() -> &'static Mutex<Counter> {
    RUNTIM_CACHE.get_or_init(|| Mutex::new(Counter { total: 0, count: 0 }))
}

struct Counter {
    total: u32,
    count: u32,
}
impl Counter {
    fn avg(&self) -> u32 {
        self.total.checked_div(self.count).unwrap_or(0)
    }
    fn add(&mut self, time: u32) {
        self.total += time;
        self.count += 1;
    }
}

struct SinkHandle {
    _handle: MixerDeviceSink,
    sink: Player,
}

static STREAM_HANDLE: OnceLock<SinkHandle> = OnceLock::new();

fn get_sink() -> &'static SinkHandle {
    STREAM_HANDLE.get_or_init(|| {
        let mut stream_handle =
            DeviceSinkBuilder::open_default_sink().expect("Failed to open default audio sink");

        stream_handle.log_on_drop(false);

        let sink = Player::connect_new(stream_handle.mixer());
        sink.set_volume(0.02);

        SinkHandle {
            _handle: stream_handle,
            sink,
        }
    })
}

pub fn main() -> iced::Result {
    iced::application(Pokedex::new, Pokedex::update, Pokedex::view)
        .theme(Theme::Dark)
        .title(Pokedex::title)
        .run()
}

#[allow(clippy::large_enum_variant)]
// possible new state changing "actions" to handle
#[derive(Debug, Clone)]
enum Message {
    NewSearch,
    PokemonFound(Result<Pokemon, Error>),
}

// state of the program
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum Pokedex {
    Loading,
    Loaded { pokemon: Pokemon },
    Errored,
}

impl Pokedex {
    // State and
    fn new() -> (Self, Task<Message>) {
        (Self::Loading, Self::search())
    }

    fn search() -> Task<Message> {
        // does async search for pokemon,
        // returns the resulting Result<Pokemon, Error> when the async future is resolved
        // wrapped in a Message::PokemonFound enum
        Task::perform(Pokemon::search(), Message::PokemonFound)
    }

    fn play_ogg_from_bytes(bytes: Option<Bytes>) {
        if let Some(ogg_bytes) = bytes
            && let Ok(source) = Decoder::new(io::Cursor::new(ogg_bytes))
        {
            let sink = &get_sink().sink;
            sink.stop();
            sink.append(source);
        }
    }

    // changes the title based on state
    fn title(&self) -> String {
        let subtitle = match self {
            Pokedex::Loading => "Loading",
            Pokedex::Loaded { pokemon } => &pokemon.name,
            Pokedex::Errored => "Whoops!",
        };

        format!("{subtitle} - Pokédex")
    }

    // update function takes in the current state and state changing action
    // the function modifies the state based on the action
    // and returns the task action to be taken
    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::PokemonFound(Ok(mut pokemon)) => {
                Self::play_ogg_from_bytes(pokemon.cry_sound_bytes.take());
                *self = Pokedex::Loaded { pokemon };

                Task::none()
            }
            Message::PokemonFound(Err(_error)) => {
                *self = Pokedex::Errored;
                Task::none()
            }
            Message::NewSearch => {
                *self = Pokedex::Loading;
                Self::search()
            }
        }
    }

    // view takes in the current state and defines how to display the current state
    // and what state changing actions each UI element can output
    fn view(&self) -> Element<'_, Message> {
        let content: Element<_> = match self {
            Pokedex::Loading => text("Searching for Pokémon...").size(40).into(),

            Pokedex::Loaded { pokemon } => column![
                pokemon.view(),
                button("Keep searching!")
                    .padding(10)
                    .on_press(Message::NewSearch)
            ]
            .max_width(500)
            .spacing(20)
            .align_x(Left)
            .into(),

            Pokedex::Errored => column![
                text("Whoops! Something went wrong...").size(40),
                button("Try again").padding(10).on_press(Message::NewSearch)
            ]
            .spacing(20)
            .align_x(Left)
            .into(),
        };

        center(content).into()
    }
}

// state of the pokemon, if we have pokemon loaded
#[derive(Debug, Clone)]
struct Pokemon {
    number: u16,
    name: String,
    description: String,
    gif_frames: Arc<Frames>,
    element_types: Vec<Handle>,
    cry_sound_bytes: Option<Bytes>,
}

impl Pokemon {
    const MAX_ID: u16 = 809;

    // pokemon view if a pokemon is found
    fn view(&self) -> Element<'_, Message> {
        fn type_image(handle: Handle) -> Image<Handle> {
            Image::new(handle).width(42.0).height(20.0)
        }

        let type_image_row = row(self
            .element_types
            .iter()
            .map(|handle| type_image(handle.clone()).into()))
        .spacing(5)
        .align_y(Bottom);

        row![
            column![
                type_image_row,
                container(Gif::new(&self.gif_frames).content_fit(iced::ContentFit::Contain))
                    .width(400.0)
                    .height(220.0)
                    .center(100.)
            ]
            .align_x(Center)
            .spacing(20),
            column![
                row![
                    text(&self.name).size(30).width(Fill),
                    text!("#{}", self.number).size(20).color([0.5, 0.5, 0.5]),
                ]
                .align_y(Center)
                .spacing(20),
                self.description.as_ref(),
            ]
            .spacing(20),
        ]
        .spacing(20)
        .align_y(Center)
        .into()
    }

    async fn search() -> Result<Pokemon, Error> {
        let now = Instant::now();
        use serde::Deserialize;
        let id = fastrand::u16(1..=Pokemon::MAX_ID);

        // -------------------------- pokemon entry struct --------------------------
        #[derive(Debug, Deserialize)]
        struct Entry {
            name: String,
            flavor_text_entries: Vec<FlavorText>,
        }

        #[derive(Debug, Deserialize)]
        struct FlavorText {
            flavor_text: String,
            language: Language,
        }

        #[derive(Debug, Deserialize)]
        struct Language {
            name: String,
        }

        // -------------------------- pokemon data struct --------------------------
        #[derive(Debug, Deserialize)]
        struct PokemonData {
            types: Vec<ElementType>,
        }

        #[derive(Debug, Deserialize)]
        struct ElementType {
            #[serde(rename = "type")]
            type_info: TypeInfo,
        }

        #[derive(Debug, Deserialize)]
        struct TypeInfo {
            name: String,
        }
        // -------------------------------------------------------------------------

        let fetch_pokemon_sprite = || async move {
            let sprite_url = format!(
                "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/showdown/{id}.gif"
            );
            client_fetch(&sprite_url).await?.bytes().await
        };

        let fetch_pokemon_cry = || async move {
            let ogg_url = format!(
                "https://raw.githubusercontent.com/PokeAPI/cries/main/cries/pokemon/latest/{id}.ogg"
            );
            client_fetch(&ogg_url).await?.bytes().await
        };

        let fetch_pokemon_entry = || async move {
            let entry_url: String = format!("https://pokeapi.co/api/v2/pokemon-species/{id}");
            client_fetch(&entry_url).await?.json::<Entry>().await
        };

        let fetch_pokemon_data = || async move {
            let data_url = format!("https://pokeapi.co/api/v2/pokemon/{id}");
            client_fetch(&data_url).await?.json::<PokemonData>().await
        };

        // Phase 1: all four independent fetches run concurrently, each retried on its own.
        let (entry, pokemon_data, frame_bytes, ogg_bytes) = futures::try_join!(
            async_retries(fetch_pokemon_entry, 4),
            async_retries(fetch_pokemon_data, 4),
            async_retries(fetch_pokemon_sprite, 4),
            async_retries(fetch_pokemon_cry, 4),
        )?;

        // Phase 2: type images depend on pokemon_data; fetch concurrently (cached after first use).
        let mut type_names: Vec<&str> = pokemon_data
            .types
            .iter()
            .map(|x| x.type_info.name.as_str())
            .collect();
        type_names.sort();

        let element_images =
            futures::future::try_join_all(type_names.into_iter().map(Self::fetch_type_images))
                .await?;

        let description = entry
            .flavor_text_entries
            .into_iter()
            .find(|t| t.language.name == "en")
            .ok_or(Error::LanguageError)?;

        let filtered_description = description
            .flavor_text
            .replace("-\n", "")
            .replace("\u{ad}\n", "")
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        let frames = Frames::from_bytes(frame_bytes.into()).map_err(|_| Error::APIError)?;

        let time_taken = now.elapsed().as_millis();
        let mut cached_time = runtime_cache().lock().unwrap();
        cached_time.add(time_taken as u32);

        println!("avg time taken: {}", cached_time.avg());

        Ok(Pokemon {
            number: id,
            name: entry.name.to_uppercase(),
            description: filtered_description,
            gif_frames: Arc::new(frames),
            element_types: element_images,
            cry_sound_bytes: Some(ogg_bytes),
        })
    }

    // for getting pokemon type IMG to display
    async fn fetch_type_images(pokemon_type: &str) -> Result<Handle, reqwest::Error> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(cache_map) = cache().lock()
                && let Some(handle) = cache_map.get(pokemon_type).cloned()
            {
                println!("getting cached image: {pokemon_type}");
                return Ok(handle);
            }
        }

        let upper_cased = {
            let mut chars = pokemon_type.chars();
            match chars.next() {
                None => unreachable!("Pokemon type shouldn't be empty"),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        };
        let url = format!("https://play.pokemonshowdown.com/sprites/types/{upper_cased}.png");

        #[cfg(not(target_arch = "wasm32"))]
        {
            let bytes: bytes::Bytes = client_fetch(&url).await?.bytes().await?;
            let type_handle = Handle::from_bytes(bytes);

            if let Ok(mut cache_map) = cache().lock() {
                cache_map.insert(pokemon_type.to_owned(), type_handle.clone());
            }

            Ok(type_handle)
        }

        #[cfg(target_arch = "wasm32")]
        Ok(Handle::from_path(url))
    }
}

async fn async_retries<F, Fut, T, E>(mut f: F, retries: u64) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut attempt = 0;
    loop {
        match f().await {
            Ok(value) => return Ok(value),
            Err(error) if attempt == retries => return Err(error),
            Err(error) => {
                eprintln!("Attempt {} failed: {:?}", attempt + 1, error);
                futures_timer::Delay::new(Duration::from_millis(300 * (attempt + 1))).await;
                attempt += 1;
            }
        }
    }
}

// Errors
#[derive(Debug, Clone)]
enum Error {
    APIError,
    LanguageError,
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Error {
        dbg!(error);
        Error::APIError
    }
}
