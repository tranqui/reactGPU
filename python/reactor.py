import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import reactgpu
import cellpolarisation as cell

import h5py, os, argparse

from progressbar import ProgressBar, Percentage, Bar, ETA
widgets = [Percentage(), ' ', Bar(), ' ', ETA()]


class ReactorTrajectory:
    def create(path, Du=1., Dv=1., k=0.08, Nx=1024, Ny=1024, dx=1., dy=1., dt=1e-2, **kwargs):
        with h5py.File(path, 'w') as stream:
            parameters = stream.attrs

            # System properties.
            parameters['Du'] = Du
            parameters['Dv'] = Dv
            parameters['k'] = k

            # Discretisation settings.
            parameters['Nx'] = Nx
            parameters['Ny'] = Ny
            parameters['dx'] = dx
            parameters['dy'] = dy
            parameters['dt'] = dt

            # Create groups for storing the fields.
            u_group = stream.create_group('u')
            v_group = stream.create_group('v')

        return ReactorTrajectory(path, mode='a', **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream: self.stream.close()
        return False

    def __init__(self, path, mode='r', **kwargs):
        if not os.path.exists(path):
            raise RuntimeError('no trajectory file {}!'.format(path))

        self.mode = mode
        self.stream = h5py.File(path, mode)
        self.kernel = None

    def close(self):
        self.stream.close()

    def start_kernel(self, u=None, v=None):
        if self.kernel is not None:
            raise RuntimeError('kernel already started!')

        if len(self.previous_steps) > 0:
            assert u is None
            assert v is None
            u, v, step = self.latest_u, self.latest_v, self.latest_step
        else:
            assert u is not None
            assert v is not None
            assert u.shape == (self.Nx, self.Ny)
            assert v.shape == (self.Nx, self.Ny)
            step = 0

        self.kernel = reactgpu.Reactor(u, v, self.dt, self.dx, self.dy, self.Du, self.Dv, self.k, step)
        return self.kernel

    @property
    def previous_steps(self):
        return np.sort([int(key) for key in self.u_group.keys()])

    @property
    def latest_step(self):
        prev = self.previous_steps
        if len(prev) > 0: return self.previous_steps[-1]
        else: return 0

    @property
    def u_group(self):
        return self.stream['u']
    @property
    def v_group(self):
        return self.stream['v']

    @property
    def latest_u(self):
        return self.u_group[str(self.latest_step)][...]
    @property
    def latest_v(self):
        return self.v_group[str(self.latest_step)][...]

    @property
    def current_step(self):
        if self.kernel is None: raise RuntimeError('kernel not yet started!')
        return self.kernel.step
    @property
    def current_u(self):
        if self.kernel is None: raise RuntimeError('kernel not yet started!')
        return self.kernel.u
    @property
    def current_v(self):
        if self.kernel is None: raise RuntimeError('kernel not yet started!')
        return self.kernel.v

    def save_current(self, compression='gzip', compression_opts=9):
        if self.kernel is None: raise RuntimeError('kernel not yet started!')
        step = self.kernel.step
        kwargs = {'compression': compression, 'compression_opts': compression_opts}
        self.u_group.create_dataset(str(step), data=self.current_u, **kwargs)
        self.v_group.create_dataset(str(step), data=self.current_v, **kwargs)

    @property
    def parameters(self):
        return self.stream.attrs

    @property
    def Du(self):
        return self.parameters['Du']
    @property
    def Dv(self):
        return self.parameters['Dv']

    @property
    def k(self):
        return self.parameters['k']

    @property
    def Nx(self):
        return self.parameters['Nx']
    @property
    def Ny(self):
        return self.parameters['Ny']

    @property
    def dx(self):
        return self.parameters['dx']
    @property
    def dy(self):
        return self.parameters['dy']
    @property
    def dt(self):
        return self.parameters['dt']

    @property
    def Lx(self):
        return self.dx * self.Nx
    @property
    def Ly(self):
        return self.dx * self.Ny

    @property
    def x(self):
        x = self.dx * np.arange(self.Nx)

    @property
    def y(self):
        y = self.dy * np.arange(self.Ny)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='reactor',
        description='Simulate a two-component reaction-diffusion model in 2d.')
    parser.add_argument('path', help='trajectory file to dump.')
    parser.add_argument('t', nargs='?', type=float)
    parser.add_argument('tfinal', nargs='?', type=float)
    parser.add_argument('-n', '--new', action='store_true')
    parser.add_argument('-m', '--movie', type=str)
    parser.add_argument('-Nx', type=int, default=1024, help='')
    parser.add_argument('-Ny', type=int, default=1024, help='')
    parser.add_argument('-dx', type=float, default=1., help='')
    parser.add_argument('-dy', type=float, default=1., help='')
    parser.add_argument('-Du', type=float, default=1., help='')
    parser.add_argument('-Dv', type=float, default=1., help='')
    parser.add_argument('-u0', type=float, help='')
    parser.add_argument('-v0', type=float, help='')
    parser.add_argument('-k', type=float, default=0.08, help='')
    parser.add_argument('-dt', type=float, default=1e-2, help='size of the timestep to use.')

    args = parser.parse_args()


    if args.new:
        with ReactorTrajectory.create(**vars(args)) as sim:
            assert (args.u0 is None) == (args.v0 is None)
            if args.u0 is None:
                eq = cell.CellPolarisationPDE(k=args.k)
                u0, v0 = eq.saddle_point

            u = np.random.random((args.Nx, args.Ny))
            v = np.random.random((args.Nx, args.Ny))
            u += u0 - np.average(u)
            v += v0 - np.average(v)

            sim.start_kernel(u, v)
            sim.save_current()


    if args.t is not None:
        if args.tfinal is None: args.tfinal = args.t
        assert args.tfinal >= args.t

        ndumps = int(np.round(args.tfinal / args.t))
        error = np.abs(ndumps - args.tfinal / args.t)
        assert np.isclose(error, 0)

        progress = ProgressBar(widgets=widgets)
        step_iterator = range(ndumps)
        if ndumps > 1: step_iterator = progress(step_iterator)

        with ReactorTrajectory(args.path, 'a') as sim:
            nsteps_per_dump = int(np.round(args.t / sim.dt))
            error = np.abs(nsteps_per_dump - args.t / sim.dt)
            assert np.isclose(error, 0)

            sim.start_kernel()
            print(f'simulating between t={sim.kernel.time} and t={sim.kernel.time + args.tfinal}:')
            for i in step_iterator:
                sim.kernel.run(nsteps_per_dump)
                sim.save_current()


    if args.movie is not None:
        with ReactorTrajectory(args.path) as sim:
            u = sim.latest_u
            steps = sim.previous_steps

            fig = plt.figure()
            ax = plt.gca()
            im = ax.imshow(u, animated=True)
            fig.colorbar(im, ax=ax, fraction=0.05, pad=2e-2)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_title('$t=0$')

            progress = ProgressBar(widgets=widgets, maxval=(len(steps)-1))
            print('rendering movie:')
            progress.start()

            def update(frame):
                u = sim.u_group[str(steps[frame])][...]
                im.set_data(u)
                ax.set_title('$t={}$'.format(steps[frame]))
                progress.update(frame)
                return im,

            anim = FuncAnimation(fig, update, frames=len(steps))
            anim.save(args.movie, writer='ffmpeg', fps=20)
            progress.finish()
