use std::{fs::File, path::PathBuf, str::FromStr};

use clap::{Parser, Subcommand};
use glam::{Vec2, Vec3};
use riptide_io::{decode::FluidDataDecoder, encode::FluidDataEncoder};
use smallvec::SmallVec;

mod run;

#[derive(Parser)]
#[command(version, about, author)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run {
        #[arg(short, long)]
        outfile: PathBuf,

        #[arg(short, long)]
        fps: u32,

        #[arg(short, long)]
        dim: u8,

        /// Size of the domain in which the fluid resides.
        #[arg(short, long, value_parser = parse_dyn_vect::<f32>, default_value = "3.0")]
        size: SmallVec<[f32; 4]>,

        /// Resolution of the grid representing the fluid.
        #[arg(long, default_value = "100")]
        res: u32,

        /// Radius of particles relative to the size of a grid cell in the fluid.
        #[arg(short, long, default_value = "0.2")]
        radius: f32,
    },
    View {
        datfile: PathBuf,
    }
}

fn parse_vect<const D: usize, T>(s: &str) -> Result<[T; D], String>
where
    T: FromStr,
    [T; D]: for<'a> TryFrom<&'a [T]>,
    for<'a> <[T; D] as TryFrom<&'a [T]>>::Error: std::fmt::Debug,
{
    let seps = s.chars().filter(|&c| c == 'x').count();
    if seps != D - 1 {
        return Err(format!("expected resolution to have exactly one separator but got {seps}."));
    }

    let mut parts = Vec::new();

    for (i, part) in s.split('x').enumerate() {
        let v: T = part.parse().map_err(|_| format!("could not parse {i}-th component of {D}-d vector `{part}`."))?;
        parts.push(v);
    }

    let res: [T; D] = parts.as_slice().try_into().unwrap();

    Ok(res)
}

fn parse_dyn_vect<T>(s: &str) -> Result<SmallVec<[T; 4]>, String>
where 
    T: FromStr,
{
    let mut parts = SmallVec::new();

    for (i, part) in s.split('x').enumerate() {
        let v: T = part.parse().map_err(|_| format!("could not parse {i}-th component of vector `{part}`."))?;
        parts.push(v);
    }

    Ok(parts)
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            outfile,
            fps,
            dim,
            mut size,
            res,
            radius
        } => {
            if size.len() == 1 {
                let v = size[0];
                for _ in 1..dim {
                    size.push(v);
                }
            }

            if size.len() as u8 != dim {
                println!("error: dimension of 'size' ({}) and the given dimension ({dim}) do not match", size.len());
                return;
            }

            let encoder = FluidDataEncoder::new(File::create(outfile).unwrap());

            if dim == 2 {
                run::run_d2(encoder, fps, Vec2::new(size[0], size[1]), res, radius);
            } else if dim == 3 {
                run::run_d3(encoder, fps, Vec3::new(size[0], size[1], size[2]), res, radius);
            }
        },
        Commands::View {
            datfile,
        } => {
            let mut decoder = FluidDataDecoder::new(File::open(datfile).unwrap());
            let data = decoder.decode().unwrap();

            if data.dim == 2 {
                riptide_view::view_2d(data);
            } else if data.dim == 3 {
                riptide_view::view_3d(data);
            }
        },
    }
}
