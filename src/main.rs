use opencv::core::{bitwise_and, Scalar, Vector, BORDER_REFLECT};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{
    adaptive_threshold, bilateral_filter, cvt_color, median_blur, ADAPTIVE_THRESH_MEAN_C,
    COLOR_BGR2GRAY, THRESH_BINARY,
};
use opencv::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let image = imread("grumpy-cat.jpg", IMREAD_COLOR)?;

    let mut gray = Mat::default();
    cvt_color(&image, &mut gray, COLOR_BGR2GRAY, 1)?;

    let mut gray_blurred = Mat::default();
    median_blur(&gray, &mut gray_blurred, 5)?;

    let mut edges = Mat::default();
    adaptive_threshold(
        &gray_blurred,
        &mut edges,
        255.0,
        ADAPTIVE_THRESH_MEAN_C,
        THRESH_BINARY,
        9,
        9.0,
    )?;

    let scalar = Scalar::from((240.0, 240.0, 240.0, 0.0));
    let mut reduced = Mat::default();
    bitwise_and(&image, &scalar, &mut reduced, &Mat::default())?;

    let mut color = Mat::default();
    bilateral_filter(&reduced, &mut color, 13, 250.0, 350.0, BORDER_REFLECT)?;

    let mut cartoon = Mat::default();
    bitwise_and(&color, &color, &mut cartoon, &edges)?;

    imwrite("output.jpg", &cartoon, &Vector::default())?;

    imshow("Cartoon", &cartoon)?;
    wait_key(0)?;

    Ok(())
}
