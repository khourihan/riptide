use std::{path::PathBuf, str::FromStr};

use clap::{Parser, Subcommand};
use riptide_io::{decode::FluidDataDecoder, encode::FluidDataEncoder};

mod run;

#[cfg(feature = "d2")]
const DIM: usize = 2;
#[cfg(feature = "d3")]
const DIM: usize = 3;

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
        outdir: PathBuf,

        #[arg(short, long)]
        fps: u32,

        #[arg(short, long)]
        time: f32,

        /// Size of the domain in which the fluid resides.
        #[arg(short, long, value_parser = parse_vect::<f32>, default_value = "3.0")]
        size: [f32; DIM],

        /// Resolution of the grid representing the fluid.
        #[arg(long, default_value = "100")]
        res: u32,

        /// Radius of particles relative to the size of a grid cell in the fluid.
        #[arg(short, long, default_value = "0.2")]
        radius: f32,
    },
    View {
        datdir: PathBuf,
    }
}

fn parse_vect<T>(s: &str) -> Result<[T; DIM], String>
where
    T: FromStr + Copy,
    [T; DIM]: for<'a> TryFrom<&'a [T]>,
    for<'a> <[T; DIM] as TryFrom<&'a [T]>>::Error: std::fmt::Debug,
{
    let seps = s.chars().filter(|&c| c == 'x').count();
    if seps == 0 {
        let res: T = s.parse().map_err(|_| format!("could not parse value `{s}`."))?;
        return Ok([res; DIM])
    }

    if seps != DIM - 1 {
        return Err(format!("expected resolution to have exactly one separator but got {seps}."));
    }

    let mut parts = Vec::new();

    for (i, part) in s.split('x').enumerate() {
        let v: T = part.parse().map_err(|_| format!("could not parse {i}-th component of {DIM}-d vector `{part}`."))?;
        parts.push(v);
    }

    let res: [T; DIM] = parts.as_slice().try_into().unwrap();

    Ok(res)
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            outdir: outfile,
            fps,
            time,
            size,
            res,
            radius
        } => {
            let frames = (time * fps as f32) as u64;
            let encoder = FluidDataEncoder::new(outfile, frames, fps).unwrap();

            #[cfg(feature = "d2")]
            run::run_d2(encoder, fps, frames, glam::Vec2::from(size), res, radius);
            #[cfg(feature = "d3")]
            run::run_d3(encoder, fps, frames, glam::Vec3::from(size), res, radius);
        },
        Commands::View {
            datdir,
        } => {
            let mut decoder = FluidDataDecoder::new(datdir);
            let meta = decoder.decode_metadata().unwrap();

            if meta.dim == 2 {
                riptide_view::view_2d(decoder, meta);
            } else if meta.dim == 3 {
                riptide_view::view_3d(decoder, meta);
            }
        },
    }
}
