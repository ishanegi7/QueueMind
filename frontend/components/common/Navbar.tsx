'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

type ApiStatus = 'connected' | 'disconnected' | 'checking';

interface NavbarProps {
  apiStatus?: ApiStatus;
}

interface NavLink {
  href: string;
  label: string;
}

const NAV_LINKS: readonly NavLink[] = [
  { href: '/', label: 'Dashboard' },
  { href: '/analytics', label: 'Analytics' },
  { href: '/simulation', label: 'Simulation' },
] as const;

const STATUS_STYLES: Record<ApiStatus, { dot: string; label: string }> = {
  connected: { dot: 'bg-green-400', label: 'API Connected' },
  disconnected: { dot: 'bg-red-400', label: 'API Disconnected' },
  checking: { dot: 'bg-amber-400 animate-pulse', label: 'Checking API' },
};

export function Navbar({ apiStatus = 'checking' }: NavbarProps) {
  const pathname = usePathname();
  const status = STATUS_STYLES[apiStatus];

  return (
    <nav className="bg-slate-900 border-b border-slate-700/50 px-6 py-3">
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        {/* Brand */}
        <div className="flex flex-col">
          <span className="text-lg font-bold text-white tracking-tight">
            QueueMind
          </span>
          <span className="text-xs text-slate-400">ED Flow Intelligence</span>
        </div>

        {/* Nav Links */}
        <div className="flex items-center gap-1">
          {NAV_LINKS.map((link) => {
            const isActive =
              link.href === '/'
                ? pathname === '/'
                : pathname.startsWith(link.href);

            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-slate-700/60 text-white'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </div>

        {/* Right: Status + Badge */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2" title={status.label}>
            <span
              className={cn('h-2 w-2 rounded-full', status.dot)}
              aria-label={status.label}
              role="status"
            />
            <span className="text-xs text-slate-400 hidden sm:inline">
              {status.label}
            </span>
          </div>
          <span className="text-xs font-medium text-slate-500 bg-slate-800 px-2 py-0.5 rounded">
            Prototype
          </span>
        </div>
      </div>
    </nav>
  );
}
