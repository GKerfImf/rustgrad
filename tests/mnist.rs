#[cfg(test)]
mod tests {

    #[cfg(test)]
    mod mnist {

        use rand::Rng;
        use mnist::*;
        use ndarray::prelude::*;

        use rustgrad::core::nonlinearity::NonLinearity::*;
        use rustgrad::mlp::layer::LayerSpec::*;
        use rustgrad::mlp::mlp::MLP;
        use rustgrad::mlp::loss::Loss;

        #[test]
        fn main() {
            // TODO: remove some duplication

            let train = 45_000;
            let dev = 5_000;
            let test = 10_000;


            let one_hot = |x: f64| -> Vec<f64> {
                let mut v = vec![0.0; 10];
                v[x as usize] = 1.0;
                v
            };

            // Deconstruct the returned Mnist struct.
            println!("Loading MNIST dataset...");
            let Mnist {
                trn_img,
                trn_lbl,
                tst_img,
                tst_lbl,
                ..
            } = MnistBuilder::new()
                .label_format_digit()
                .training_set_length(50_000)
                .validation_set_length(10_000)
                .test_set_length(10_000)
                .finalize();

            // Can use an Array2 or Array3 here (Array3 for visualization)
            let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
                .expect("Error converting images to Array3 struct")
                .map(|x| *x as f64 / 256.0);

            // Convert the returned Mnist struct to Array2 format
            let train_labels: Array2<f64> = Array2::from_shape_vec((50_000, 1), trn_lbl)
                .expect("Error converting training labels to Array2 struct")
                .map(|x| *x as f64);

            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
            //                          Initialize and train the model                         //
            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //

            // Prepare the training data
            let x_train =
                (0..train).map( |image_num|
                    train_data.slice(s![image_num, .., ..])
                    .iter().map( |x| *x as f64 )
                    .collect::<Vec<f64>>()
                ).collect::<Vec<Vec<f64>>>();

            let y_train =
                (0..train).map( |image_num|
                    train_labels.slice(s![image_num, ..])
                    .iter().flat_map( |x| one_hot(*x) )
                    .collect::<Vec<f64>>()
                ).collect::<Vec<Vec<f64>>>();

            println!("Constructing the model...");
            let mlp = MLP::new(
                784, vec![
                    FullyConnected(16), NonLinear(ReLu),
                    FullyConnected(10)
                ]
            );
            let loss = Loss::with_multi_class_hinge_loss(&mlp);

            println!("Training the model...");
            for i in 0..10000 {
                println!("Loss = {:.8}, \t Iter = {}", loss.rand_batch_train(&mlp, &x_train, &y_train, 64, 0.05), i);
            }

            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
            //                              Evaluate on dev data                               //
            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
            // Prepare dev data
            let x_dev =
                (train..train+dev).map( |image_num|
                    train_data.slice(s![image_num, .., ..])
                    .iter().map( |x| *x as f64 )
                    .collect::<Vec<f64>>()
                ).collect::<Vec<Vec<f64>>>();

            let y_dev =
                (train..train+dev).map( |image_num|
                    train_labels.slice(s![image_num, ..])
                    .iter().flat_map( |x| one_hot(*x) )
                    .collect::<Vec<f64>>()
                ).collect::<Vec<Vec<f64>>>();

            let right_predictions = (0..dev).map( |i| {
                let prediction : usize = mlp.eval(&x_dev[i]).iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                let label : usize = y_dev[i].iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                (prediction == label) as i32
            });
            println!("--------------------------------Accuracy on dev data--------------------------------");
            println!("Accuracy: {}", right_predictions.sum::<i32>() as f64 / dev as f64);

            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
            //                              Run on the test data                               //
            // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //

            // let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
            //     .expect("Error converting images to Array3 struct")
            //     .map(|x| *x as f32 / 256.);

            // let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
            //     .expect("Error converting testing labels to Array2 struct")
            //     .map(|x| *x as f32);

            // let x_test =
            //     (0..test).map( |image_num|
            //         test_data.slice(s![image_num, .., ..])
            //         .iter().map( |x| *x as f64 )
            //         .collect::<Vec<f64>>()
            //     ).collect::<Vec<Vec<f64>>>();

            // let y_test =
            //     (0..test).map( |image_num|
            //         test_labels.slice(s![image_num, ..])
            //         .iter().map( |x| *x as f64 )
            //         .collect::<Vec<f64>>()
            //     ).collect::<Vec<Vec<f64>>>();

            // println!("-------------------------------Accuracy on test data--------------------------------");
            // for _ in 0..25 {
            //     let i = rand::thread_rng().gen_range(0..test);
            //     let prediction : usize = mlp.eval(&x_train[i]).iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            //     let label : usize = y_train[i].iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            //     println!("Model: {:?}, \tLabel: {:?}, \tMatch: {:?}", prediction, label, prediction == label);
            // }

        }
    }
}