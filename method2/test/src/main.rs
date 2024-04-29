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

static THRESHOLD: f64 = 0.0001;

fn main() {
    // Load .npy files into ndarray arrays
    let array1: ndarray::Array3<u8> =
        ndarray::Array3::<u8>::read_npy(File::open("../../reference_matrixes.npy").unwrap())
            .unwrap();
    let array1 = array1.slice(s![1500..2000, .., ..]).map(|&x| x as i16);
    let array2: ndarray::Array3<u8> =
        ndarray::Array3::<u8>::read_npy(File::open("../../subject_matrixes.npy").unwrap()).unwrap();
    let array2 = array2.map(|&x| x as i16);
    println!("Done Computing");

    let mut true_positive = 0.0;
    let mut false_positive = 0.0;
    let mut false_negative = 0.0;
    let mut true_negative = 0.0;
    let out = array1
        .axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .map(|(i, reference)| {
            let mut inner_true_positive = 0.0;
            let mut inner_false_positive = 0.0;
            let mut inner_false_negative = 0.0;
            let mut inner_true_negative = 0.0;
            for j in 1500..2000 {
                //println!("j = {}", j);
                let diff = &reference.view() - &array2.index_axis(ndarray::Axis(0), j).view();
                //println!("Diff {:?}", diff);
                //println!("Chom {:?}", diff.mapv(|x| x * x));
                //println!("Sum {:?}", diff.mapv(|x| x * x).sum());
                let mse = (diff.mapv(|x| x * x).sum() as f64 / (reference.len() as f64)).abs();
                //println!("MSE: {}", mse);
                if mse < THRESHOLD {
                    if i == j {
                        inner_true_positive += 1.0;
                    } else {
                        inner_false_positive += 1.0;
                    }
                } else {
                    if i == j {
                        inner_false_negative += 1.0;
                    } else {
                        inner_true_negative += 1.0;
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
        .collect::<Vec<(f32, f32, f32, f32)>>();
    for (tp, fp, _fn, tn) in out {
        true_positive += tp;
        false_positive += fp;
        false_negative += _fn;
        true_negative += tn;
    }
    println!("True Positive: {:?}", true_positive);
    println!("True Negative: {:?}", true_negative);
    println!("False Positive: {:?}", false_positive);
    println!("False Negative: {:?}", false_negative);
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
        2.0 * (true_positive / (true_positive + false_positive))
            * (true_positive / (true_positive + false_negative))
            / ((true_positive / (true_positive + false_positive))
                + (true_positive / (true_positive + false_negative)))
    );
    println!(
        "Accuracy: {}",
        (true_positive + true_negative)
            / (true_positive + true_negative + false_positive + false_negative)
    );
}
