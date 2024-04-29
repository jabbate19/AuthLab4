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
use itertools::min;
use itertools::max;

static START: f32 = 0.0;
static END: f32 = 1.0;
static SAMPLES: u16 = 100;
static STEP: f32 = (END - START) / SAMPLES as f32;

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
            .slice(s![..1500, .., ..])
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
            .slice(s![..1500, .., ..])
            .map(|&x| x as i16)
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| median(x.to_owned()))
            .collect::<Vec<_>>(),
    );
    println!("{}", array2.len());
    println!("Done Computing");
    let mut thresholds = Vec::new();
    let mut false_positives = Vec::new();
    let mut false_negatives = Vec::new();
    let mut true_positives = Vec::new();
    let mut true_negatives = Vec::new();
    for threshold in (0..SAMPLES).map(|x| START as f32 + (x as f32) * STEP) {
        println!("Threshold: {}", threshold);
        let mut true_positive = 0;
        let mut false_positive = 0;
        let mut false_negative = 0;
        let mut true_negative = 0;
        for i in 0..1500 {
            for j in 0..1500 {
                let pce = ((array1[i] - array2[j]) / array1[i]).abs();
                if pce < threshold {
                    if i == j {
                        true_positive += 1;
                    } else {
                        false_positive += 1;
                    }
                } else {
                    if i == j {
                        false_negative += 1;
                    } else {
                        true_negative += 1;
                    }
                }
            }
        }
        thresholds.push(threshold);
        false_positives.push(false_positive);
        false_negatives.push(false_negative);
        true_positives.push(true_positive);
        true_negatives.push(true_negative);
    }
    println!("Thresholds: {:?}", thresholds);
    println!("False Positives: {:?}", false_positives);
    println!("False Negatives: {:?}", false_negatives);
    println!("True Positives: {:?}", true_positives);
    println!("True Negatives: {:?}", true_negatives);
    let mut last_threshold = 0.0;
    for i in 0..thresholds.len() {
        if false_negatives[i] < false_positives[i] {
            println!("Threshold: {}", (thresholds[i] + last_threshold) / 2.0);
            break;
        }
        last_threshold = thresholds[i];
    }
    let far_min = min(false_positives.iter().cloned());
    let frr_min = min(false_negatives.iter().cloned());
    let far_max = max(false_positives.iter().cloned());
    let frr_max = max(false_negatives.iter().cloned());
    let far_avg = false_positives.iter().sum::<u32>() as f32  / false_positives.len() as f32 / (1500.0*1500.0);
    let frr_avg = false_negatives.iter().sum::<u32>() as f32 / false_negatives.len() as f32 / (1500.0*1500.0);
    println!("FAR Min: {}%", far_min.unwrap() as f32 / (1500.0*1500.0) * 100.0);
    println!("FRR Min: {}%", frr_min.unwrap() as f32 / (1500.0*1500.0) * 100.0);
    println!("FAR Max: {}%", far_max.unwrap() as f32 / (1500.0*1500.0) * 100.0);
    println!("FRR Max: {}%", frr_max.unwrap() as f32 / (1500.0*1500.0) * 100.0);
    println!("FAR Avg: {}%", far_avg * 100.0);
    println!("FRR Avg: {}%", frr_avg * 100.0);
    println!("EER: {}%", (far_avg + frr_avg) / 2.0 * 100.0);
}
