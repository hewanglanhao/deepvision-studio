import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('@shell/home/home-page.component')
      .then(m => m.HomePageComponent)
  },
  {
    path: 'login',
    loadComponent: () => import('@core/auth/auth-page.component')
      .then(m => m.AuthPageComponent),
    data: { mode: 'login' }
  },
  {
    path: 'register',
    loadComponent: () => import('@core/auth/auth-page.component')
      .then(m => m.AuthPageComponent),
    data: { mode: 'register' }
  },
  {
    path: 'mode-a',
    loadComponent: () => import('@modes/mode-a/mode-a-page.component')
      .then(m => m.ModeAPageComponent),
    data: { mode: 'forward' }
  },
  {
    path: 'teaching',
    loadComponent: () => import('@shell/teaching/teaching-doc-page.component')
      .then(m => m.TeachingDocPageComponent)
  },
  {
    path: 'mode-b',
    loadComponent: () => import('@modes/mode-b/mode-b-page.component')
      .then(m => m.ModeBPageComponent),
    data: { mode: 'training' }
  },
  {
    path: 'training/experiments',
    loadComponent: () => import('@modes/mode-b/experiment-compare/experiment-compare-page.component')
      .then(m => m.ExperimentComparePageComponent)
  },
  {
    path: 'training/inference',
    loadComponent: () => import('@modes/mode-b/single-inference/single-inference-page.component')
      .then(m => m.SingleInferencePageComponent)
  },
  {
    path: 'training/collaboration',
    loadComponent: () => import('@modes/mode-b/training-collaboration/training-collaboration-page.component')
      .then(m => m.TrainingCollaborationPageComponent)
  },
  {
    path: 'mode-c',
    loadComponent: () => import('@modes/mode-c/mode-c-page.component')
      .then(m => m.ModeCPageComponent)
  },
  {
    path: 'mode-d',
    loadComponent: () => import('@modes/mode-d/mode-d-page.component')
      .then(m => m.ModeDPageComponent)
  },
  {
    path: 'mode-e',
    loadComponent: () => import('@modes/mode-e/mode-e-page.component')
      .then(m => m.ModeEPageComponent)
  },
  {
    path: 'mode-f',
    loadComponent: () => import('@modes/mode-f/mode-f-page.component')
      .then(m => m.ModeFPageComponent)
  },
  {
    path: 'ai-museum',
    loadComponent: () => import('@modes/ai-museum/ai-museum-page.component')
      .then(m => m.AiMuseumPageComponent)
  },
  {
    path: 'network-3d',
    loadComponent: () => import('@shared/network-3d/network-3d-viewer.component')
      .then(m => m.Network3dViewerComponent)
  },
  { path: 'forward', redirectTo: 'mode-a', pathMatch: 'full' },
  { path: 'training', redirectTo: 'mode-b', pathMatch: 'full' },
  { path: '**', redirectTo: '' }
];
