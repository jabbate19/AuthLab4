use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::s;
use ndarray::ArrayBase;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray_npy::ReadNpyExt;
use rayon::prelude::*;
use std::fs::File;
use std::time::Instant;
use itertools::min;
use itertools::max;

static START: f32 = 0.0;
static END: f32 = 1.0;
static SAMPLES: u16 = 100;
static STEP: f32 = (END - START) / SAMPLES as f32;

fn main() {
    // Load .npy files into ndarray arrays
    let array1: ndarray::Array3<u8> =
        ndarray::Array3::<u8>::read_npy(File::open("../../reference_matrixes.npy").unwrap())
            .unwrap();
    let array1 = array1.slice(s![1500..2000, .., ..]).map(|&x| x as i16);
    let array2: ndarray::Array3<u8> =
        ndarray::Array3::<u8>::read_npy(File::open("../../subject_matrixes.npy").unwrap()).unwrap();
    let array2 = array2.slice(s![1500..2000, .., ..]).map(|&x| x as i16);
    println!("Done Computing");
    let mut thresholds = Vec::new();
    let mut false_positives = Vec::new();
    let mut false_negatives = Vec::new();
    for threshold in (0..SAMPLES).map(|x| START as f32 + (x as f32) * STEP) {
        println!("Threshold: {}", threshold);
        let mut true_positive = 0;
        let mut false_positive = 0;
        let mut false_negative = 0;
        let mut true_negative = 0;
        let out = array1
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .map(|(i, reference)| {
                let mut inner_true_positive = 0;
                let mut inner_false_positive = 0;
                let mut inner_false_negative = 0;
                let mut inner_true_negative = 0;
                for j in 0..500 {
                    //println!("j = {}", j);
                    let diff = &reference.view() - &array2.index_axis(ndarray::Axis(0), j).view();
                    //println!("Diff {:?}", diff);
                    //println!("Chom {:?}", diff.mapv(|x| x * x));
                    //println!("Sum {:?}", diff.mapv(|x| x * x).sum());
                    let mse = (diff.mapv(|x| x * x).sum() as f64 / (reference.len() as f64)).abs();
                    //println!("MSE: {}", mse);
                    if mse < threshold.into() {
                        if i == j {
                            inner_true_positive += 1;
                        } else {
                            inner_false_positive += 1;
                        }
                    } else {
                        if i == j {
                            inner_false_negative += 1;
                        } else {
                            inner_true_negative += 1;
                        }
                    }
                }
                (
                    inner_true_positive,
                    inner_false_positive,
                    inner_false_negative,
                    inner_true_negative,
                )
            })
            .collect::<Vec<(u32, u32, u32, u32)>>();
        for (tp, fp, _fn, tn) in out {
            true_positive += tp;
            false_positive += fp;
            false_negative += _fn;
            true_negative += tn;
        }
        false_negatives.push(false_negative);
        false_positives.push(false_positive);
        thresholds.push(threshold);
    }
    println!("Thresholds: {:?}", thresholds);
    println!("False Positives: {:?}", false_positives);
    println!("False Negatives: {:?}", false_negatives);
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
    println!("FAR Min: {}%", far_min.unwrap() as f32 / (500.0*500.0) * 100.0);
    println!("FRR Min: {}%", frr_min.unwrap() as f32 / (500.0*500.0) * 100.0);
    println!("FAR Max: {}%", far_max.unwrap() as f32 / (500.0*500.0) * 100.0);
    println!("FRR Max: {}%", frr_max.unwrap() as f32 / (500.0*500.0) * 100.0);
    println!("FAR Avg: {}%", far_avg * 100.0);
    println!("FRR Avg: {}%", frr_avg * 100.0);
    println!("EER: {}%", (far_avg + frr_avg) / 2.0 * 100.0);
}
