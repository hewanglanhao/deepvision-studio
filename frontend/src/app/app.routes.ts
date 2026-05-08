import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./pages/home/home-page.component')
      .then(m => m.HomePageComponent)
  },
  {
    path: 'login',
    loadComponent: () => import('./pages/auth/auth-page.component')
      .then(m => m.AuthPageComponent),
    data: { mode: 'login' }
  },
  {
    path: 'register',
    loadComponent: () => import('./pages/auth/auth-page.component')
      .then(m => m.AuthPageComponent),
    data: { mode: 'register' }
  },
  {
    path: 'mode-a',
    loadComponent: () => import('./pages/mode-a/mode-a-page.component')
      .then(m => m.ModeAPageComponent),
    data: { mode: 'forward' }
  },
  {
    path: 'mode-b',
    loadComponent: () => import('./pages/mode-b/mode-b-page.component')
      .then(m => m.ModeBPageComponent),
    data: { mode: 'training' }
  },
  {
    path: 'mode-c',
    loadComponent: () => import('./pages/mode-c/mode-c-page.component')
      .then(m => m.ModeCPageComponent)
  },
  {
    path: 'mode-d',
    loadComponent: () => import('./pages/mode-d/mode-d-page.component')
      .then(m => m.ModeDPageComponent)
  },
  {
    path: 'mode-e',
    loadComponent: () => import('./pages/mode-e/mode-e-page.component')
      .then(m => m.ModeEPageComponent)
  },
  {
    path: 'network-3d',
    loadComponent: () => import('./features/network-3d/network-3d-viewer.component')
      .then(m => m.Network3dViewerComponent)
  },
  { path: 'forward', redirectTo: 'mode-a', pathMatch: 'full' },
  { path: 'training', redirectTo: 'mode-b', pathMatch: 'full' },
  { path: '**', redirectTo: '' }
];
