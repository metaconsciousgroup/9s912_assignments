import os
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cpu')

BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_T = 30
FOV = 32

TRAIN_SAVES_DIR = 'train_saves'
SAVED_MODELS_DIR = 'saved_models'

version = 'version_string'

if not os.path.exists(TRAIN_SAVES_DIR):
    os.makedirs(TRAIN_SAVES_DIR)
if not os.path.exists(SAVED_MODELS_DIR):
    os.makedirs(SAVED_MODELS_DIR)


uniform_fov = torch.linspace(0, 2 * np.pi, FOV+1)[:-1]


def generate_batch(batch_size):
    thetas = torch.rand(batch_size) * 2 * np.pi
    coords = torch.meshgrid(thetas, uniform_fov)
    sads = torch.abs(torch.atan2(torch.sin(coords[0] - coords[1]), torch.cos(coords[0] - coords[1]))).to(device)
    inputs = torch.cos(sads)
    centers = torch.zeros(batch_size) + np.pi
    coords2 = torch.meshgrid(centers, uniform_fov)
    targets = torch.cos(torch.abs(torch.atan2(torch.sin(coords2[0] - coords2[1]), torch.cos(coords2[0] - coords2[1]))))
    return inputs, targets


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear(FOV * 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, curr_fov, init_fov):
        x = F.tanh(self.fc1(torch.hstack((curr_fov, init_fov))))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


def shift_fov(curr_ang, shift):
    new_ang = curr_ang + shift.T[0]
    coords = torch.meshgrid(new_ang, uniform_fov)
    sads = torch.abs(torch.atan2(torch.sin(coords[0] - coords[1]), torch.cos(coords[0] - coords[1]))).to(device)
    new_fov = torch.cos(sads)
    return new_ang, new_fov


def run_batch(action, rnn, inputs, targets, rnn_optimizer=None):
    """Train or test on a single batch."""
    with torch.set_grad_enabled(action == 'train'):
        if action == 'train':
            rnn_optimizer.zero_grad()
        loss = 0

        curr_fov = inputs
        curr_ang = torch.linspace(0, 2 * np.pi, BATCH_SIZE + 1)[:-1]

        fovs = [curr_fov.detach().numpy()]
        angles = [curr_ang.detach().numpy()]

        for _ in range(NUM_T):
            shift = rnn(curr_fov, inputs.detach())
            shift += torch.normal(0, 0.1, (BATCH_SIZE, 1))
            curr_ang, curr_fov = shift_fov(curr_ang, shift)

            fovs.append(curr_fov.detach().numpy())
            angles.append(curr_ang.detach().numpy())

            loss += torch.mean(torch.square(targets - curr_fov))

        if action == 'train':
            loss.backward()
            rnn_optimizer.step()

        return loss.item(), np.array(fovs), np.array(angles)


def as_minutes(seconds):
    return '{:d}m {:d}s'.format(math.floor(seconds / 60), int(seconds - 60 * math.floor(seconds / 60)))


def train(rnn, version, learning_rate, batch_size, max_iters=5e6, print_every=200, test_every=1e13, save_every=1e13):
    train_losses, test_losses = [], []
    loss_buffer = 0

    rnn_optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

    print('Training model version: {}'.format(version))
    start_time = time.time()
    for curr_iter in range(1, int(max_iters) + 1):
        inputs, targets = generate_batch(batch_size)

        train_loss, fovs, angles = run_batch('train', rnn, inputs, targets, rnn_optimizer)
        loss_buffer += train_loss
        train_losses.append(train_loss)

        if curr_iter % print_every == 0:
            loss_avg = loss_buffer / print_every
            loss_buffer = 0
            time_elapsed = as_minutes(time.time() - start_time)
            print('{} ({:d} {:d}%) {:.4f}'.format(time_elapsed, curr_iter, round(curr_iter / max_iters * 100), loss_avg))

        if curr_iter % test_every == 0:
            test_loss, _, _, _ = run_batch('test', rnn, test_inputs, test_targets)
            test_losses.append(test_loss)
            print('Current Test Loss: {:.3f}'.format(test_loss))

        if curr_iter % save_every == 0:
            with open('{}/train_losses-{}.pickle'.format(TRAIN_SAVES_DIR, version), 'wb') as f:
                pickle.dump(train_losses, f)
            with open('{}/val_losses-{}.pickle'.format(TRAIN_SAVES_DIR, version), 'wb') as f:
                pickle.dump(test_losses, f)
            torch.save(rnn.state_dict(), '{}/rnn-{}-it{}.pt'.format(SAVED_MODELS_DIR, version, curr_iter))


rnn = RNN()
train(rnn, version, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)

########################################################################

test_inputs, test_targets = generate_batch(BATCH_SIZE)
loss, fovs, angles = run_batch('test', rnn, test_inputs, test_targets)

plt.title('Landmark attraction by RNN')
plt.xlabel('Time steps')
plt.ylabel('Angle of landmark in visual field')
plt.plot(angles[:, np.all(angles < 8, axis=0)]-np.pi)

# (time steps + 1, batch size, field of view)
plt.title('Time 0')
plt.xlabel('Field of view index')
plt.ylabel('Trial #')
plt.imshow(fovs[0, :, :], cmap='Greys_r')

plt.title('Time 30')
plt.xlabel('Field of view index')
plt.ylabel('Trial #')
plt.imshow(fovs[30, :, :], cmap='Greys_r')
