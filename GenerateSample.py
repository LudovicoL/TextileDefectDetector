import torch
from torch import nn

import backbone as b
from config import *

def main():
    model = b.VariationalAutoencoder(latent_space, learning_rate).to(device)
    model.load_state_dict(torch.load(outputs_dir + 'state_dict'))

    image_sample = model.sample(1, device)
    image_generated = model.generate(image_sample)

    b.plot_couple(image_sample[0][0], image_generated[0][0], b.assemble_pathname("Reconstructed sample"), histogram=False)

    original_histogram = torch.histc(image_sample)
    generated_histogram = torch.histc(image_generated)

    b.plot_couple(original_histogram, generated_histogram, b.assemble_pathname("Histogram sample"), histogram=True)

    loss = nn.MSELoss()
    output = loss(image_sample, image_generated)
    print('MSE between two histograms: ' + str(output.item()))

if __name__ == '__main__':
    main()