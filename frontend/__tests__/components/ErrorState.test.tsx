import React from 'react';
import { render, screen } from '@testing-library/react';
import { ErrorState } from '../../components/common/ErrorState';

describe('ErrorState', () => {
  it('renders generic error by default', () => {
    render(<ErrorState message="Something went wrong" />);
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('renders connection error', () => {
    render(<ErrorState type="connection" message="API is unreachable" />);
    expect(screen.getByText('QueueMind API is unavailable')).toBeInTheDocument();
    expect(screen.getByText('API is unreachable')).toBeInTheDocument();
  });
});
