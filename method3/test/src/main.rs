use itertools::Itertools;
use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::s;
use ndarray::Array;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray_npy::ReadNpyExt;
use rayon::prelude::*;
use std::fs::File;
use std::iter::FromIterator;
use std::time::Instant;

static THRESHOLD: f32 = 0.005;

fn median(arr: Array2<i16>) -> f32 {
    let mut arr = Array::from_iter(arr.iter().sorted().cloned());
    let len = arr.len();
    if len % 2 == 0 {
        (arr[len / 2] + arr[len / 2 - 1]) as f32 / 2.0
    } else {
        arr[len / 2] as f32
    }
}

fn main() {
    // Load .npy files into ndarray arrays
    println!("Computing Medians");
    let array1: ndarray::Array3<u8> =
        ndarray::Array3::<u8>::read_npy(File::open("../../reference_matrixes.npy").unwrap())
            .unwrap();
    let mut array1 = Array::from_iter(
        array1
            .slice(s![1500..2000, .., ..])
            .map(|&x| x as i16)
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| median(x.to_owned()))
            .collect::<Vec<_>>(),
    );
    println!("{}", array1.len());
    let array2: ndarray::Array3<u8> =
        ndarray::Array3::<u8>::read_npy(File::open("../../subject_matrixes.npy").unwrap()).unwrap();
    let array2 = Array::from_iter(
        array2
            .slice(s![1500..2000, .., ..])
            .map(|&x| x as i16)
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| median(x.to_owned()))
            .collect::<Vec<_>>(),
    );
    println!("{}", array2.len());
    println!("Done Computing");
    let mut true_positive = 0.0;
    let mut false_positive = 0.0;
    let mut false_negative = 0.0;
    let mut true_negative = 0.0;
    for i in 0..500 {
        for j in 0..500 {
            let pce = ((array1[i] - array2[j]) / array1[i]).abs();
            if pce < THRESHOLD {
                if i == j {
                    true_positive += 1.0;
                } else {
                    false_positive += 1.0;
                }
            } else {
                if i == j {
                    false_negative += 1.0;
                } else {
                    true_negative += 1.0;
                }
            }
        }
    }
    println!("True Positive: {}", true_positive);
    println!("False Positive: {}", false_positive);
    println!("False Negative: {}", false_negative);
    println!("True Negative: {}", true_negative);
    println!(
        "Accuracy: {}",
        (true_positive + true_negative)
            / (true_positive + true_negative + false_positive + false_negative)
    );
    println!(
        "Precision: {}",
        true_positive / (true_positive + false_positive)
    );
    println!(
        "Recall: {}",
        true_positive / (true_positive + false_negative)
    );
    println!(
        "F1 Score: {}",
        2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative)
    );
}
