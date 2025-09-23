import { Navigation, MobileNavigation } from "./Navigation";

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Mobile Navigation */}
      <MobileNavigation />

      {/* Desktop Navigation */}
      <div className="hidden lg:block">
        <Navigation />
      </div>

      {/* Main Content */}
      <main className="flex-1 overflow-x-hidden">
        <div className="h-full">
          {children}
        </div>
      </main>
    </div>
  );
}