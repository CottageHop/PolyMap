export interface PolyMapOptions {
  center?: { lat: number; lon: number };
  bbox?: { south: number; west: number; north: number; east: number };
  zoom?: number;
  tilt?: number;
  dataUrl?: string;
  wasmUrl?: string;
  layers?: Partial<{
    buildings: boolean;
    roads: boolean;
    water: boolean;
    parks: boolean;
    trees: boolean;
    shadows: boolean;
    labels: boolean;
  }>;
}

export interface MarkerOptions {
  className?: string;
  html?: string;
}

export interface CameraState {
  lat: number;
  lon: number;
  zoom: number;
  tilt: number;
}

export interface MarkerClickEvent {
  id: string;
  lat: number;
  lon: number;
  element: HTMLElement;
}

export type PolyMapEvent =
  | 'ready'
  | 'camera:move'
  | 'click'
  | 'resize'
  | 'marker:click';

export interface PolyMapInstance {
  setView(lat: number, lon: number, zoom?: number): void;
  setZoom(zoom: number): void;
  setTilt(tilt: number): void;
  panBy(dx: number, dy: number): void;
  getCamera(): CameraState;

  addMarker(id: string, lat: number, lon: number, options?: MarkerOptions): HTMLElement;
  removeMarker(id: string): void;
  clearMarkers(): void;
  getMarkerElement(id: string): HTMLElement | null;

  setLayerVisible(layer: string, visible: boolean): void;

  on(event: 'ready', callback: (map: PolyMapInstance) => void): PolyMapInstance;
  on(event: 'camera:move', callback: (camera: CameraState) => void): PolyMapInstance;
  on(event: 'marker:click', callback: (data: MarkerClickEvent) => void): PolyMapInstance;
  on(event: PolyMapEvent, callback: (...args: any[]) => void): PolyMapInstance;
  off(event: string, callback?: Function): PolyMapInstance;

  destroy(): void;
}

export function initWasm(wasmUrl?: string): Promise<any>;
export function createPolyMap(container: string | HTMLElement, options?: PolyMapOptions): Promise<PolyMapInstance>;
export default createPolyMap;
