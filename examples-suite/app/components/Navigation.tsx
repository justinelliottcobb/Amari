import { Link, useLocation } from "@remix-run/react";
import { Button, H3, Card, CardBody } from "jadis-ui";

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
      }
    ]
  }
];

export function Navigation() {
  const location = useLocation();

  return (
    <nav className="w-80 h-screen overflow-y-auto bg-background border-r border-border p-6">
      <div className="mb-8">
        <Link to="/" className="block">
          <H3>Amari Library</H3>
          <p className="text-sm text-muted-foreground">Mathematical Computing Examples</p>
        </Link>
      </div>

      <div className="space-y-6">
        {navigationSections.map((section) => (
          <div key={section.title}>
            <h4 className="font-semibold text-sm uppercase tracking-wider text-muted-foreground mb-3">
              {section.title}
            </h4>
            <div className="space-y-2">
              {section.items.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <Card key={item.href} className={`transition-all duration-200 ${isActive ? 'ring-2 ring-primary' : 'hover:shadow-md'}`}>
                    <CardBody className="p-3">
                      <Link to={item.href} className="block">
                        <div className="font-medium text-sm mb-1">{item.title}</div>
                        <div className="text-xs text-muted-foreground">{item.description}</div>
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
    <div className="lg:hidden bg-background border-b border-border p-4">
      <details className="group">
        <summary className="flex items-center justify-between cursor-pointer">
          <H3>Navigation</H3>
          <span className="transition-transform group-open:rotate-180">
            ▼
          </span>
        </summary>
        <div className="mt-4 space-y-4">
          {navigationSections.map((section) => (
            <div key={section.title}>
              <h4 className="font-semibold text-sm uppercase tracking-wider text-muted-foreground mb-2">
                {section.title}
              </h4>
              <div className="grid grid-cols-1 gap-2">
                {section.items.map((item) => (
                  <Button key={item.href} href={item.href} className="text-left justify-start">
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