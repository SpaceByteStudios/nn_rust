#![allow(dead_code)]

use nn_rust::neural_net::data_point::DataPoint;

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

    let root = BitMapBackend::new(&path, (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let (left, right) = root.split_horizontally(400);

    let img_area = left.margin(40, 40, 40, 40);

    let mut chart = ChartBuilder::on(&img_area)
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

    let y: &DataPoint = data_point;
    let mut max_y_index = 0;

    for i in 1..y.exp_output.len() {
        if y.exp_output.get(i) > y.exp_output.get(max_y_index) {
            max_y_index = i;
        }
    }

    let style =
        FontDesc::new(FontFamily::SansSerif, 20.0, FontStyle::Normal).into_text_style(&left);

    left.draw_text(&format!("Label: {}", max_y_index), &style, (160, 360))?;

    let bar_width: i32 = 30;
    let spacing: i32 = 10;
    let max_height: i32 = 350;

    for i in 0..10 {
        let p = prediction.exp_output.get(i);
        let pct = (p * 100.0) as i32;

        let x0 = (i as i32) * (bar_width + spacing);
        let x1 = x0 + bar_width;

        let bar_height = pct * (max_height - 50) / 100;
        let y0 = max_height - bar_height;
        let y1 = max_height;

        // bar
        right.draw(&Rectangle::new([(x0, y0), (x1, y1)], RED.filled()))?;

        // label
        right.draw(&Text::new(
            format!("{i} : {pct}%"),
            (x0 - 2, max_height + 10),
            ("sans-serif", 12),
        ))?;
    }

    root.present()?;
    Ok(())
}
