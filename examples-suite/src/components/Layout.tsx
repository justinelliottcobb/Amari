import { AppShell, Burger, ScrollArea } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { Navigation } from './Navigation'

interface LayoutProps {
  children: React.ReactNode
}

export function Layout({ children }: LayoutProps) {
  const [opened, { toggle }] = useDisclosure()

  return (
    <AppShell
      header={{ height: { base: 60, md: 0 } }}
      navbar={{
        width: 300,
        breakpoint: 'md',
        collapsed: { mobile: !opened },
      }}
      padding="md"
    >
      <AppShell.Header hiddenFrom="md">
        <Burger
          opened={opened}
          onClick={toggle}
          size="sm"
          p="md"
          aria-label="Toggle navigation"
        />
      </AppShell.Header>

      <AppShell.Navbar>
        <AppShell.Section grow component={ScrollArea}>
          <Navigation onNavigate={() => opened && toggle()} />
        </AppShell.Section>
      </AppShell.Navbar>

      <AppShell.Main>
        {children}
      </AppShell.Main>
    </AppShell>
  )
}
