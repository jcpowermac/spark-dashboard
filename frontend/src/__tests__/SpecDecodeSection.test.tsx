import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { SpecDecodeSection } from '@/components/engines/EnginePanelPrimitives'

const base = {
  acceptanceRate: 80,
  acceptanceRateLive: null,
  meanAcceptanceLength: 4,
  acceptedTokens: 800,
  draftTokens: 1000,
}

describe('SpecDecodeSection per-position acceptance', () => {
  it('renders one bar per draft position when the vector is present', () => {
    render(
      <SpecDecodeSection
        {...base}
        acceptedTokensPerPos={[800, 400, 120]}
      />,
    )
    expect(screen.getByTitle('pos 0: 800')).toBeInTheDocument()
    expect(screen.getByTitle('pos 1: 400')).toBeInTheDocument()
    expect(screen.getByTitle('pos 2: 120')).toBeInTheDocument()
  })

  it('omits the strip when the engine reports no per-position vector', () => {
    const { container } = render(<SpecDecodeSection {...base} />)
    expect(container.querySelectorAll('[title^="pos "]')).toHaveLength(0)
  })
})
