#![windows_subsystem = "windows"]

use iced::{
    Bottom, Center, Element, Fill, Left, Task, Theme, futures,
    widget::{
        button, center, column, container,
        image::{Handle, Image},
        row, text,
    },
};

use std::{sync::Arc, time::Duration};

use iced_gif::{Frames, Gif};
use rodio::{Decoder, OutputStream, Sink};

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

    fn play_ogg_from_bytes(option_bytes: Option<Vec<u8>>) {
        std::thread::spawn(move || {
            if let Some(ogg_bytes) = option_bytes
                && let Ok((_stream, stream_handle)) = OutputStream::try_default()
                && let Ok(sink) = Sink::try_new(&stream_handle)
                && let Ok(source) = Decoder::new(std::io::Cursor::new(ogg_bytes))
            {
                sink.append(source);
                sink.set_volume(0.02);
                sink.sleep_until_end();
            }
        });
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
    cry_sound_bytes: Option<Vec<u8>>,
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
        use rand::Rng;
        use serde::Deserialize;

        let id = {
            let mut rng = rand::rngs::OsRng;
            rng.gen_range(1..=Pokemon::MAX_ID)
        };

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
                        reqwest::get(&url).await?.json::<Entry>().await
                    }
                })
            };

            let fetch_pokemon_images = || {
                futures::FutureExt::boxed(async move {
                    let url = format!("https://pokeapi.co/api/v2/pokemon/{id}");
                    let mut pokemon_data = reqwest::get(&url).await?.json::<PokemonData>().await?;

                    pokemon_data
                        .types
                        .sort_by(|a, b| a.type_info.name.cmp(&b.type_info.name));

                    let image_futures: Vec<_> = pokemon_data
                        .types
                        .into_iter()
                        .map(|t| Self::fetch_type_images(t.type_info.name))
                        .collect();

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
                let resp = reqwest::get(&url).await?;
                let bytes = resp.bytes().await?;
                Ok(bytes.to_vec())
            };

            futures::future::try_join4(
                async_retries(fetch_pokemon_entry, 4),
                async_retries(fetch_pokemon_images, 4),
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
    async fn fetch_pokemon_image(id: u16) -> Result<Vec<u8>, reqwest::Error> {
        let url = format!(
            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/showdown/{id}.gif"
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            let bytes = reqwest::get(&url).await?.bytes().await?;

            Ok(bytes.to_vec())
        }

        #[cfg(target_arch = "wasm32")]
        Ok(Handle::from_path(url))
    }

    // for getting pokemon type IMG to display
    async fn fetch_type_images(pokemon_type: String) -> Result<Handle, reqwest::Error> {
        let upper_cased = {
            let mut chars = pokemon_type.chars();
            match chars.next() {
                None => String::new(), // empty string
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        };
        let url = format!("https://play.pokemonshowdown.com/sprites/types/{upper_cased}.png");

        #[cfg(not(target_arch = "wasm32"))]
        {
            let bytes = reqwest::get(&url).await?.bytes().await?;

            Ok(Handle::from_bytes(bytes))
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

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Error {
        dbg!(error);
        Error::APIError
    }
}
