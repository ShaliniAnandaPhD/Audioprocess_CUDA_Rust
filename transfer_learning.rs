use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::sync::Arc;
use tch::{nn, Device, Tensor};

// Define the base model for transfer learning
struct BaseModel {
    model: nn::Sequential,
}

impl BaseModel {
    fn new(input_size: i64, hidden_size: i64, output_size: i64) -> Self {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = nn::seq()
            .add(nn::linear(&vs.root(), input_size, hidden_size, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.root(), hidden_size, output_size, Default::default()));
        BaseModel { model }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.model.forward(input)
    }
}

// Define the target model for transfer learning
struct TargetModel {
    base_model: BaseModel,
    fc: nn::Linear,
}

impl TargetModel {
    fn new(base_model: BaseModel, num_classes: i64) -> Self {
        let vs = nn::VarStore::new(Device::Cpu);
        let fc = nn::linear(&vs.root(), base_model.model.variables().last().unwrap().size()[0], num_classes, Default::default());
        TargetModel { base_model, fc }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let features = self.base_model.forward(input);
        self.fc.forward(&features)
    }

    fn freeze_base_model(&mut self) {
        for param in self.base_model.model.variables() {
            param.set_requires_grad(false);
        }
    }

    fn unfreeze_base_model(&mut self) {
        for param in self.base_model.model.variables() {
            param.set_requires_grad(true);
        }
    }
}

// Function to train the base model
fn train_base_model(
    model: &BaseModel,
    data: &[f32],
    labels: &[i64],
    epochs: i64,
    batch_size: i64,
    learning_rate: f64,
) {
    let data_tensor = Tensor::of_slice(data).reshape(&[-1, data.len() as i64 / labels.len()]);
    let labels_tensor = Tensor::of_slice(labels).to_device(Device::Cpu);
    let dataset = data_tensor.to_device(Device::Cpu);

    let mut optimizer = nn::Adam::default().build(&model.model.vars(), learning_rate).unwrap();

    let num_samples = dataset.size()[0];
    let num_batches = num_samples / batch_size;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        let shuffled_indices = Tensor::randperm(num_samples, (Kind::Int64, Device::Cpu));

        for i in 0..num_batches {
            let offset = i * batch_size;
            let batch_indices = shuffled_indices.narrow(0, offset, batch_size);
            let batch = dataset.index_select(0, &batch_indices);
            let batch_labels = labels_tensor.index_select(0, &batch_indices);

            let output = model.forward(&batch);
            let loss = output.cross_entropy_for_logits(&batch_labels);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.double_value(&[]);
        }

        let avg_loss = epoch_loss / num_batches as f64;
        println!("Epoch [{}/{}], Base Model Loss: {:.4}", epoch + 1, epochs, avg_loss);
    }
}

// Function to train the target model
fn train_target_model(
    model: &mut TargetModel,
    data: &[f32],
    labels: &[i64],
    epochs: i64,
    batch_size: i64,
    learning_rate: f64,
    fine_tune: bool,
) {
    let data_tensor = Tensor::of_slice(data).reshape(&[-1, data.len() as i64 / labels.len()]);
    let labels_tensor = Tensor::of_slice(labels).to_device(Device::Cpu);
    let dataset = data_tensor.to_device(Device::Cpu);

    let mut optimizer = nn::Adam::default().build(&model.fc.variables(), learning_rate).unwrap();

    let num_samples = dataset.size()[0];
    let num_batches = num_samples / batch_size;

    model.freeze_base_model();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        let shuffled_indices = Tensor::randperm(num_samples, (Kind::Int64, Device::Cpu));

        for i in 0..num_batches {
            let offset = i * batch_size;
            let batch_indices = shuffled_indices.narrow(0, offset, batch_size);
            let batch = dataset.index_select(0, &batch_indices);
            let batch_labels = labels_tensor.index_select(0, &batch_indices);

            let output = model.forward(&batch);
            let loss = output.cross_entropy_for_logits(&batch_labels);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.double_value(&[]);
        }

        let avg_loss = epoch_loss / num_batches as f64;
        println!("Epoch [{}/{}], Target Model Loss: {:.4}", epoch + 1, epochs, avg_loss);
    }

    if fine_tune {
        model.unfreeze_base_model();

        let mut optimizer = nn::Adam::default().build(&model.base_model.model.vars(), learning_rate).unwrap();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            let shuffled_indices = Tensor::randperm(num_samples, (Kind::Int64, Device::Cpu));

            for i in 0..num_batches {
                let offset = i * batch_size;
                let batch_indices = shuffled_indices.narrow(0, offset, batch_size);
                let batch = dataset.index_select(0, &batch_indices);
                let batch_labels = labels_tensor.index_select(0, &batch_indices);

                let output = model.forward(&batch);
                let loss = output.cross_entropy_for_logits(&batch_labels);

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                epoch_loss += loss.double_value(&[]);
            }

            let avg_loss = epoch_loss / num_batches as f64;
            println!("Epoch [{}/{}], Fine-tuned Model Loss: {:.4}", epoch + 1, epochs, avg_loss);
        }
    }
}

// Python module to expose transfer learning functions
#[pymodule]
fn transfer_learning(_py: Python, m: &PyModule) -> PyResult<()> {
    // Function to train the base model
    #[pyfn(m)]
    #[pyo3(name = "train_base_model")]
    fn train_base_model_py(
        input_size: i64,
        hidden_size: i64,
        output_size: i64,
        data: Vec<f32>,
        labels: Vec<i64>,
        epochs: i64,
        batch_size: i64,
        learning_rate: f64,
    ) -> PyResult<BaseModel> {
        let model = BaseModel::new(input_size, hidden_size, output_size);
        train_base_model(&model, &data, &labels, epochs, batch_size, learning_rate);
        Ok(model)
    }

    // Function to train the target model
    #[pyfn(m)]
    #[pyo3(name = "train_target_model")]
    fn train_target_model_py(
        base_model: BaseModel,
        num_classes: i64,
        data: Vec<f32>,
        labels: Vec<i64>,
        epochs: i64,
        batch_size: i64,
        learning_rate: f64,
        fine_tune: bool,
    ) -> PyResult<TargetModel> {
        let mut model = TargetModel::new(base_model, num_classes);
        train_target_model(&mut model, &data, &labels, epochs, batch_size, learning_rate, fine_tune);
        Ok(model)
    }

    Ok(())
}