import React from 'react';
import { render, screen } from '@testing-library/react';
import { QueueHealthHero } from '../../components/dashboard/QueueHealthHero';
import type { QueueHealthResponse } from '../../lib/types';

const mockData: QueueHealthResponse = {
  score: 85,
  state: 'CRITICAL',
  components: {
    congestion_pressure: 90,
    arrival_pressure: 80,
    high_acuity_pressure: 60,
  },
  weights: {
    congestion: 0.5,
    arrivals: 0.3,
    acuity: 0.2,
  },
  dominant_factor: 'congestion_pressure',
  summary: 'Queue health is CRITICAL.',
  non_clinical_disclaimer: 'Disclaimer test',
};

describe('QueueHealthHero', () => {
  it('renders the score and state', () => {
    render(<QueueHealthHero data={mockData} />);
    expect(screen.getByText('85')).toBeInTheDocument();
    expect(screen.getByText('/ 100')).toBeInTheDocument();
    expect(screen.getByText('CRITICAL')).toBeInTheDocument();
    expect(screen.getByText(/Primary pressure:/)).toBeInTheDocument();
  });
});
