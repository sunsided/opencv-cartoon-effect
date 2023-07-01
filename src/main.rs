use opencv::core::{
    bitwise_and, merge, split, Point, Scalar, Size, TermCriteria, Vector, BORDER_REFLECT, CV_8UC1,
};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{
    adaptive_threshold, circle, cvt_color, dilate, get_structuring_element, median_blur,
    pyr_mean_shift_filtering, resize, COLOR_BGR2Lab, COLOR_Lab2BGR, ADAPTIVE_THRESH_MEAN_C, FILLED,
    INTER_CUBIC, INTER_NEAREST, LINE_AA, MORPH_RECT, THRESH_BINARY,
};
use opencv::prelude::*;
use opencv::ximgproc::anisotropic_diffusion;
use std::error::Error;

/// An image in BGR color space.
struct Bgr(Mat);

/// An image in Lab color space.
struct Lab(Mat);

/// A grayscale / lightness / luminance image.
struct Gray(Mat);

fn main() -> Result<(), Box<dyn Error>> {
    let image = Bgr(imread("me.jpeg", IMREAD_COLOR)?);
    imshow("Original", &image.0)?;

    let lab_image = bgr_to_lab(image)?;
    let segmented_image = segment_colors(&lab_image)?;
    let image_blurred = anisotropic_blur(&lab_image)?;
    let gray = gray_from_lab(&image_blurred)?;
    let dilated_edges = get_edges(gray)?;
    let image = lab_to_bgr(segmented_image)?;
    // let image = halftone(image)?;
    let cartoon = combine_image_and_edges(image, dilated_edges)?;

    imshow("Cartoon", &cartoon)?;
    imwrite("output.jpg", &cartoon, &Vector::default())?;

    wait_key(0)?;
    Ok(())
}

/// Converts the BGR image to Lab.
fn bgr_to_lab(Bgr(image): Bgr) -> Result<Lab, Box<dyn Error>> {
    let mut lab_image = Mat::default();
    cvt_color(&image, &mut lab_image, COLOR_BGR2Lab, 0)?;
    Ok(Lab(lab_image))
}

/// Converts an Lab image back to BGR color space.
fn lab_to_bgr(Lab(image): Lab) -> Result<Bgr, Box<dyn Error>> {
    let mut bgr = Mat::default();
    cvt_color(&image, &mut bgr, COLOR_Lab2BGR, 0)?;
    Ok(Bgr(bgr))
}

/// Extracts the lightness channel from the Lab image.
fn gray_from_lab(Lab(image_blurred): &Lab) -> Result<Gray, Box<dyn Error>> {
    let mut channels = Vector::<Mat>::new();
    split(&image_blurred, &mut channels)?;

    // Extract the L channel (index 0) from the Lab image
    let gray = channels.get(0)?.clone();
    Ok(Gray(gray))
}

/// Obtains edges from the specified grayscale image.
fn get_edges(Gray(gray): Gray) -> Result<Gray, Box<dyn Error>> {
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

    // Dilate the edges, i.e. make them less prominent.
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
    Ok(Gray(dilated_edges))
}

/// Segments the colors of an Lab image.
fn segment_colors(Lab(lab_image): &Lab) -> Result<Lab, Box<dyn Error>> {
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

    Ok(Lab(segmented_image))
}

/// Applies anisotropic blurring to an Lab image.
fn anisotropic_blur(Lab(lab_image): &Lab) -> Result<Lab, Box<dyn Error>> {
    let mut image_blurred = Mat::default();
    let conductance = 0.1;
    let time_step = 0.05;
    let num_iterations = 10;
    anisotropic_diffusion(
        &lab_image,
        &mut image_blurred,
        time_step,
        conductance,
        num_iterations,
    )?;
    Ok(Lab(image_blurred))
}

/// Applies an halftone color effect.
#[allow(dead_code)]
fn halftone(Bgr(image): Bgr) -> Result<Bgr, Box<dyn Error>> {
    // Using CMYK would be more true to the cause but we'll just fake it.
    let mut channels = Vector::<Mat>::new();
    split(&image, &mut channels)?;

    for channel_idx in 0..3 {
        let channel = channels.get(channel_idx)?;
        let mut filtered = Mat::default();
        median_blur(&channel, &mut filtered, 7)?;

        // We upsample the channel so that we can operate "sub-pixel".
        let mut resized = Mat::default();
        resize(
            &filtered,
            &mut resized,
            image.size()? * 2,
            0.0,
            0.0,
            INTER_NEAREST,
        )?;

        let mut canvas = Mat::new_rows_cols_with_default(
            resized.rows(),
            resized.cols(),
            CV_8UC1,
            Scalar::default(),
        )?;

        let cols = resized.cols() as usize;
        let rows = resized.rows() as usize;

        // The grids should be rotated; for simplicity, we simply offset them.
        for y in ((3 + channel_idx)..cols).step_by(7) {
            for x in ((3 + channel_idx)..rows).step_by(7) {
                let pixel: f32 = *resized.at_2d::<u8>(x as _, y as _)? as _;
                let intensity = pixel / 255.0;
                let radius = intensity * 7.5;
                debug_assert!(radius >= 0.0 && radius <= 255.0);

                let center = Point::new(y as _, x as _);
                let color = Scalar::from(intensity.sqrt() as f64 * 255.0);
                circle(
                    &mut canvas,
                    center,
                    radius as i32,
                    color,
                    FILLED,
                    LINE_AA,
                    0,
                )?;
            }
        }

        let mut resized = Mat::default();
        resize(&canvas, &mut resized, image.size()?, 0.0, 0.0, INTER_CUBIC)?;

        channels.set(channel_idx, resized)?;
    }

    let mut halftoned = Mat::default();
    merge(&channels, &mut halftoned)?;

    Ok(Bgr(halftoned))
}

/// Applies the detected edges as brush strokes to the provided image.
fn combine_image_and_edges(
    Bgr(image): Bgr,
    Gray(dilated_edges): Gray,
) -> Result<Mat, Box<dyn Error>> {
    let mut cartoon = Mat::default();
    bitwise_and(&image, &image, &mut cartoon, &dilated_edges)?;
    Ok(cartoon)
}
