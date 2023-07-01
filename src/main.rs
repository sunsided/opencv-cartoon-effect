use opencv::core::{bitwise_and, split, Point, Scalar, Size, TermCriteria, Vector, BORDER_REFLECT};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{
    adaptive_threshold, cvt_color, dilate, get_structuring_element, pyr_mean_shift_filtering,
    COLOR_BGR2Lab, COLOR_Lab2BGR, ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C, MORPH_RECT,
    THRESH_BINARY,
};
use opencv::prelude::*;
use opencv::ximgproc::anisotropic_diffusion;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut image = imread("grumpy-cat.jpg", IMREAD_COLOR)?;

    let mut image_blurred = Mat::default();
    let conductance = 0.1;
    let time_step = 0.1;
    let num_iterations = 10;
    anisotropic_diffusion(
        &image,
        &mut image_blurred,
        time_step,
        conductance,
        num_iterations,
    )?;

    let mut lab_image = Mat::default();
    cvt_color(&image_blurred, &mut lab_image, COLOR_BGR2Lab, 0)?;

    let gray = gray_from_lab(&lab_image)?;

    let mut lab_image = Mat::default();
    cvt_color(&image, &mut lab_image, COLOR_BGR2Lab, 0)?;

    let spatial_radius = 10.0;
    let color_radius = 20.0;
    let max_pyramid_level = 1;
    let term_criteria = TermCriteria::default()?;
    let mut segmented_image = Mat::default();
    pyr_mean_shift_filtering(
        &lab_image,
        &mut segmented_image,
        spatial_radius,
        color_radius,
        max_pyramid_level,
        term_criteria,
    )?;

    cvt_color(&segmented_image, &mut image, COLOR_Lab2BGR, 0)?;
    imshow("segmented_image (BGR)", &image)?;

    let max_binary_value = 255.0;
    let mut edges = Mat::default();
    adaptive_threshold(
        &gray,
        &mut edges,
        max_binary_value,
        ADAPTIVE_THRESH_MEAN_C,
        THRESH_BINARY,
        9,
        9.0,
    )?;

    let mut dilated_edges = Mat::default();
    let kernel = get_structuring_element(MORPH_RECT, Size::new(3, 3), Point::new(-1, -1))?;
    let anchor = Point::new(-1, -1);
    let iterations = 1;
    dilate(
        &edges,
        &mut dilated_edges,
        &kernel,
        anchor,
        iterations,
        BORDER_REFLECT,
        Scalar::default(),
    )?;
    imshow("dilated_edges", &dilated_edges)?;

    let scalar = Scalar::from((240.0, 240.0, 240.0, 0.0));
    let mut reduced = Mat::default();
    bitwise_and(&image, &scalar, &mut reduced, &Mat::default())?;

    let mut cartoon = Mat::default();
    bitwise_and(&image, &image, &mut cartoon, &dilated_edges)?;

    imwrite("output.jpg", &cartoon, &Vector::default())?;

    imshow("Cartoon", &cartoon)?;

    wait_key(0)?;

    Ok(())
}

fn gray_from_lab(image_blurred: &Mat) -> Result<Mat, Box<dyn Error>> {
    let mut channels = Vector::<Mat>::new();
    split(&image_blurred, &mut channels)?;

    // Extract the L channel (index 0) from the Lab image
    let gray = channels.get(0)?.clone();
    Ok(gray)
}
