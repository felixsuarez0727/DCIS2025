import React from 'react';
import { createRoot } from 'react-dom/client';
import ModelArchitecture from './model_visualization.jsx';

const container = document.getElementById('app');
const root = createRoot(container);
root.render(<ModelArchitecture />);