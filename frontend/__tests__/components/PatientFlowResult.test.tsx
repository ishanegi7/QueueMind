import React from 'react';
import { render, screen } from '@testing-library/react';
import { PatientFlowResult } from '../../components/patient-flow/PatientFlowResult';
import type { PatientFlowResponse } from '../../lib/types';

const mockResponse: PatientFlowResponse = {
  predicted_remaining_time_minutes: 150.5,
  unit: 'minutes',
  model_name: 'xgboost',
  model_version: '1.0',
  prediction_interval: {
    lower_minutes: 100,
    upper_minutes: 200,
    coverage_level: 0.9,
    method: 'split_conformal',
    non_negative_enforced: true,
  },
  explanation: null,
  non_causal_disclaimer: 'Not causal',
};

describe('PatientFlowResult', () => {
  it('renders the formatted time correctly', () => {
    render(<PatientFlowResult result={mockResponse} />);
    expect(screen.getByText('2h 31m')).toBeInTheDocument();
    expect(screen.getByText(/Likely between/)).toBeInTheDocument();
    expect(screen.getByText('1h 40m')).toBeInTheDocument();
    expect(screen.getByText('3h 20m')).toBeInTheDocument();
  });
});
