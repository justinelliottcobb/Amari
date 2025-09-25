import { Link, useLocation } from "react-router-dom";
import { Button, H3, Card, CardBody, P, Strong, Grid, GridItem, Navbar, NavbarNav } from "jadis-ui";

interface NavSection {
  title: string;
  items: NavItem[];
}

interface NavItem {
  title: string;
  href: string;
  description: string;
}

const navigationSections: NavSection[] = [
  {
    title: "Core Mathematics",
    items: [
      {
        title: "Geometric Algebra",
        href: "/geometric-algebra",
        description: "Multivectors, geometric products, and rotors"
      },
      {
        title: "Tropical Algebra",
        href: "/tropical-algebra",
        description: "Max-plus semiring operations"
      },
      {
        title: "Dual Numbers",
        href: "/dual-numbers",
        description: "Automatic differentiation"
      },
      {
        title: "Information Geometry",
        href: "/information-geometry",
        description: "Statistical manifolds and Fisher metrics"
      }
    ]
  },
  {
    title: "Advanced Systems",
    items: [
      {
        title: "WebGPU Acceleration",
        href: "/webgpu",
        description: "GPU-accelerated computations"
      },
      {
        title: "Fusion System",
        href: "/fusion",
        description: "TropicalDualClifford integration"
      },
      {
        title: "Cellular Automata",
        href: "/automata",
        description: "Self-assembling systems"
      }
    ]
  },
  {
    title: "Tools & Playground",
    items: [
      {
        title: "Interactive Playground",
        href: "/playground",
        description: "Live code examples and experimentation"
      },
      {
        title: "Performance Benchmarks",
        href: "/benchmarks",
        description: "Compare algorithm performance"
      },
      {
        title: "API Reference",
        href: "/api-reference",
        description: "Complete function documentation"
      },
      {
        title: "Amari-Chentsov Tensor",
        href: "/amari-chentsov-tensor",
        description: "Interactive tensor visualization"
      }
    ]
  }
];

export function Navigation() {
  const location = useLocation();

  return (
    <nav style={{ height: '100vh', overflowY: 'auto', padding: '1.5rem' }}>
      <div style={{ marginBottom: '2rem' }}>
        <Link to="/">
          <H3>Amari Library</H3>
          <P>Mathematical Computing Examples</P>
        </Link>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {navigationSections.map((section) => (
          <div key={section.title}>
            <Strong style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.75rem', display: 'block' }}>
              {section.title}
            </Strong>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {section.items.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <Card key={item.href} style={{
                    transition: 'all 0.2s',
                    ...(isActive ? { outline: '2px solid var(--primary)' } : {})
                  }}>
                    <CardBody style={{ padding: '0.75rem' }}>
                      <Link to={item.href}>
                        <div style={{ fontWeight: 500, fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                          {item.title}
                        </div>
                        <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>
                          {item.description}
                        </div>
                      </Link>
                    </CardBody>
                  </Card>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </nav>
  );
}

export function MobileNavigation() {
  return (
    <div style={{ padding: '1rem', borderBottom: '1px solid var(--border)' }}>
      <details>
        <summary style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}>
          <H3>Navigation</H3>
          <span style={{ transition: 'transform 0.2s' }}>
            ▼
          </span>
        </summary>
        <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {navigationSections.map((section) => (
            <div key={section.title}>
              <Strong style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.5rem', display: 'block' }}>
                {section.title}
              </Strong>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {section.items.map((item) => (
                  <Button key={item.href} href={item.href} style={{ textAlign: 'left', justifyContent: 'flex-start' }}>
                    {item.title}
                  </Button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}