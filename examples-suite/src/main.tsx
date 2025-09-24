import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import { ThemeProvider } from 'jadis-ui'
// CSS import will be added once we resolve the path issue

import App from './App'
import { GeometricAlgebra } from './pages/GeometricAlgebra'
import { TropicalAlgebra } from './pages/TropicalAlgebra'
import { DualNumbers } from './pages/DualNumbers'
import { InformationGeometry } from './pages/InformationGeometry'
import { WebGPU } from './pages/WebGPU'
import { Fusion } from './pages/Fusion'
import { Automata } from './pages/Automata'
import { Playground } from './pages/Playground'
import { Benchmarks } from './pages/Benchmarks'
import { APIReference } from './pages/APIReference'
import { Home } from './pages/Home'

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      {
        index: true,
        element: <Home />,
      },
      {
        path: "geometric-algebra",
        element: <GeometricAlgebra />,
      },
      {
        path: "tropical-algebra",
        element: <TropicalAlgebra />,
      },
      {
        path: "dual-numbers",
        element: <DualNumbers />,
      },
      {
        path: "information-geometry",
        element: <InformationGeometry />,
      },
      {
        path: "webgpu",
        element: <WebGPU />,
      },
      {
        path: "fusion",
        element: <Fusion />,
      },
      {
        path: "automata",
        element: <Automata />,
      },
      {
        path: "playground",
        element: <Playground />,
      },
      {
        path: "benchmarks",
        element: <Benchmarks />,
      },
      {
        path: "api-reference",
        element: <APIReference />,
      },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider defaultTheme="terminal">
      <RouterProvider router={router} />
    </ThemeProvider>
  </React.StrictMode>,
)