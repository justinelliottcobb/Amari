import { describe, it, expect } from 'vitest';
import { render, screen } from '../test/test-utils';
import { Probabilistic } from './Probabilistic';

describe('Probabilistic Page', () => {
  it('renders the page title', () => {
    render(<Probabilistic />);

    expect(screen.getByRole('heading', { name: 'Probabilistic Computing' })).toBeInTheDocument();
  });

  it('renders the page description', () => {
    render(<Probabilistic />);

    const descElements = screen.getAllByText(/probability distributions/i);
    expect(descElements.length).toBeGreaterThan(0);
  });

  it('renders Gaussian distribution section', () => {
    render(<Probabilistic />);

    // Use getAllByText since "Gaussian" appears multiple times
    const gaussianElements = screen.getAllByText(/Gaussian/i);
    expect(gaussianElements.length).toBeGreaterThan(0);
  });

  it('renders MCMC section', () => {
    render(<Probabilistic />);

    const mcmcElements = screen.getAllByText(/MCMC/i);
    expect(mcmcElements.length).toBeGreaterThan(0);
  });

  it('renders stochastic process section', () => {
    render(<Probabilistic />);

    const brownianElements = screen.getAllByText(/Brownian/i);
    expect(brownianElements.length).toBeGreaterThan(0);
  });

  it('renders code examples', () => {
    render(<Probabilistic />);

    expect(screen.getAllByText(/Code:/i).length).toBeGreaterThan(0);
  });
});
