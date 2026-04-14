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

use rand::RngExt;

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

static CLIENT_CACHE: OnceLock<surf::Client> = OnceLock::new();

fn client_cache() -> &'static surf::Client {
    CLIENT_CACHE.get_or_init(surf::Client::new)
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
        self.total / self.count
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

#[derive(Debug, Clone)]
enum PokemonTypes {
    Single { type_1: Handle },
    Double { type_1: Handle, type_2: Handle },
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

    fn play_ogg_from_bytes(option_bytes: Option<Bytes>) {
        if let Some(ogg_bytes) = option_bytes
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
    element_types: PokemonTypes,
    cry_sound_bytes: Option<Bytes>,
}

impl Pokemon {
    const MAX_ID: u16 = 809;

    // pokemon view if a pokemon is found
    fn view(&self) -> Element<'_, Message> {
        fn type_image(handle: Handle) -> Image<Handle> {
            Image::new(handle).width(42.0).height(20.0)
        }

        let type_image_row = match &self.element_types {
            PokemonTypes::Single { type_1 } => row![type_image(type_1.clone())],
            PokemonTypes::Double { type_1, type_2 } => {
                row![type_image(type_1.clone()), type_image(type_2.clone())]
            }
        };

        row![
            column![
                type_image_row.spacing(5).align_y(Bottom),
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

        let id = rand::rng().random_range(1..=Pokemon::MAX_ID);

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

        println!("getting id: {id}");

        let (entry, element_images, frame_bytes, ogg_bytes) = {
            let fetch_pokemon_entry = || {
                futures::FutureExt::boxed(async move {
                    {
                        let url = format!("https://pokeapi.co/api/v2/pokemon-species/{id}");
                        client_cache().get(&url).recv_json::<Entry>().await
                    }
                })
            };

            let fetch_element_images = || {
                futures::FutureExt::boxed(async move {
                    let url = format!("https://pokeapi.co/api/v2/pokemon/{id}");
                    let pokemon_data = client_cache().get(&url).recv_json::<PokemonData>().await?;

                    let mut type_names: Vec<&str> = pokemon_data
                        .types
                        .iter()
                        .map(|x| x.type_info.name.as_str())
                        .collect();

                    type_names.sort();

                    let image_futures = type_names.into_iter().map(Self::fetch_type_images);

                    let mut element_images: Vec<Handle> =
                        futures::future::try_join_all(image_futures).await?;

                    match element_images.len() {
                        1 => Ok(PokemonTypes::Single {
                            type_1: element_images.remove(0),
                        }),
                        2 => Ok(PokemonTypes::Double {
                            type_1: element_images.remove(0),
                            type_2: element_images.remove(0),
                        }),
                        _ => unreachable!("Pokémons have 1 or 2 types"),
                    }
                })
            };

            let fetch_ogg = async {
                let url = format!(
                    "https://raw.githubusercontent.com/PokeAPI/cries/main/cries/pokemon/latest/{id}.ogg"
                );
                let bytes: bytes::Bytes = client_cache().get(&url).recv_bytes().await?.into();
                Ok(bytes)
            };

            futures::future::try_join4(
                async_retries(fetch_pokemon_entry, 4),
                async_retries(fetch_element_images, 4),
                Self::fetch_pokemon_image(id),
                fetch_ogg,
            )
            .await?
        };

        let filtered_description = {
            let description = entry
                .flavor_text_entries
                .iter()
                .find(|text| text.language.name == "en")
                .ok_or(Error::LanguageError)?;

            description
                .flavor_text
                .replace("-\n", "")
                .replace("\u{ad}\n", "")
                .chars()
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect()
        };

        let frames = Frames::from_bytes(frame_bytes).map_err(|_| Error::APIError)?;

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

    // for getting pokemon IMG to display
    async fn fetch_pokemon_image(id: u16) -> Result<Vec<u8>, surf::Error> {
        let url = format!(
            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/showdown/{id}.gif"
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            let bytes: bytes::Bytes = client_cache().get(&url).recv_bytes().await?.into();

            Ok(bytes.to_vec())
        }

        #[cfg(target_arch = "wasm32")]
        Ok(Handle::from_path(url))
    }

    // for getting pokemon type IMG to display
    async fn fetch_type_images(pokemon_type: &str) -> Result<Handle, surf::Error> {
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
            let bytes: bytes::Bytes = client_cache().get(&url).recv_bytes().await?.into();
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

async fn async_retries<F, T, E>(mut f: F, retries: u64) -> Result<T, E>
where
    F: FnMut() -> futures::future::BoxFuture<'static, Result<T, E>>,
    E: std::fmt::Debug,
{
    for attempt in 0..=retries {
        let result = f().await;

        match result {
            Ok(v) => return Ok(v),
            Err(e) => {
                if attempt == retries {
                    // return failure
                    return Err(e);
                } else {
                    // retry
                    eprintln!("Attempt {} failed: {:?}", attempt + 1, e);
                    futures_timer::Delay::new(Duration::from_millis(300 * (attempt + 1))).await;
                }
            }
        }
    }
    unreachable!()
}

// Errors
#[derive(Debug, Clone)]
enum Error {
    APIError,
    LanguageError,
}

impl From<surf::Error> for Error {
    fn from(error: surf::Error) -> Error {
        dbg!(error);
        Error::APIError
    }
}
