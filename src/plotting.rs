use crate::neural_net::data_point::DataPoint;

use plotters::prelude::*;
use plotters::style::full_palette::LIGHTBLUE;

pub fn plot_performance(
    performance: Vec<f64>,
    path: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(&path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min: f64 = 0.0;
    let y_max: f64 = performance
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .margin(30)
        .caption("Neural Network Performance", ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..performance.len() as i32, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Epochs")
        .y_desc("Cost")
        .draw()?;

    chart.draw_series(LineSeries::new(
        performance.iter().enumerate().map(|(i, y)| (i as i32, *y)),
        &RED,
    ))?;

    root.present()?;

    Ok(())
}

pub fn plot_2d_graph(
    train_data: &[DataPoint],
    predictions: &[DataPoint],
    path: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(&path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut x_min: f64 = 0.0;
    let mut x_max: f64 = 0.0;

    let mut y_min: f64 = 0.0;
    let mut y_max: f64 = 0.0;

    for t in train_data {
        if t.input.get(0) < x_min {
            x_min = t.input.get(0);
        }

        if t.input.get(0) > x_max {
            x_max = t.input.get(0);
        }

        if t.exp_output.get(0) < y_min {
            y_min = t.exp_output.get(0);
        }

        if t.exp_output.get(0) > y_max {
            y_max = t.exp_output.get(0);
        }
    }

    let mut chart = ChartBuilder::on(&root)
        .caption("Neural Network Predictions", ("sans-serif", 20))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - 0.25)..(x_max + 0.25),
            (y_min - 0.25)..(y_max + 0.25),
        )?;

    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    chart.draw_series(
        train_data
            .iter()
            .map(|x| Circle::new((x.input.get(0), x.exp_output.get(0)), 5, RED.filled())),
    )?;

    chart.draw_series(
        predictions
            .iter()
            .map(|x| Circle::new((x.input.get(0), x.exp_output.get(0)), 5, BLUE.filled())),
    )?;

    root.present()?;

    Ok(())
}

pub fn plot_2d_classification(
    train_data: &[DataPoint],
    predictions: &[DataPoint],
    path: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(&path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut x_min: f64 = 0.0;
    let mut x_max: f64 = 0.0;

    let mut y_min: f64 = 0.0;
    let mut y_max: f64 = 0.0;

    for t in train_data {
        if t.input.get(0) < x_min {
            x_min = t.input.get(0);
        }

        if t.input.get(0) > x_max {
            x_max = t.input.get(0);
        }

        if t.input.get(1) < y_min {
            y_min = t.input.get(1);
        }

        if t.input.get(1) > y_max {
            y_max = t.input.get(1);
        }
    }

    let mut chart = ChartBuilder::on(&root)
        .caption("Neural Network Predictions", ("sans-serif", 20))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - 0.25)..(x_max + 0.25),
            (y_min - 0.25)..(y_max + 0.25),
        )?;

    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    chart.draw_series(train_data.iter().map(|x| {
        Circle::new(
            (x.input.get(0), x.input.get(1)),
            5,
            if x.exp_output.get(0) > x.exp_output.get(1) {
                BLUE.filled()
            } else {
                RED.filled()
            },
        )
    }))?;

    chart.draw_series(predictions.iter().map(|x| {
        Circle::new(
            (x.input.get(0), x.input.get(1)),
            2,
            if x.exp_output.get(0) > x.exp_output.get(1) {
                LIGHTBLUE.filled()
            } else {
                YELLOW.filled()
            },
        )
    }))?;

    root.present()?;

    Ok(())
}

pub fn plot_mnist(
    data_point: &DataPoint,
    prediction: &DataPoint,
    path: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let size = 28;

    let root = BitMapBackend::new(&path, (280, 280)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(0..size, 0..size)?;

    chart.configure_mesh().disable_mesh().draw()?;

    for y in 0..size {
        for x in 0..size {
            let value = data_point.input.get((y * size + x) as usize);
            let pixel = (value * 255.0) as u8;

            let color = RGBColor(pixel, pixel, pixel);

            let flipped_y = size - y - 1;

            chart.draw_series(std::iter::once(Rectangle::new(
                [(x, flipped_y), (x + 1, flipped_y + 1)],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}
