import { readFileSync } from 'node:fs';
import { extname } from 'node:path';

export type OpenAIImageContentPart = {
  type: 'image_url';
  image_url: {
    url: string;
  };
};

export interface ImageOptions {
  url?: string | null;
  path?: string | null;
  base64_data?: string | null;
  media_type?: string | null;
}

const MEDIA_TYPE_BY_EXT: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.svg': 'image/svg+xml',
};

export function _guess_media_type(path: string): string {
  return MEDIA_TYPE_BY_EXT[extname(path).toLowerCase()] ?? 'image/png';
}

export class Image {
  readonly url: string | null;
  readonly path: string | null;
  readonly base64_data: string | null;
  readonly media_type: string | null;

  constructor(opts: ImageOptions) {
    this.url = opts.url ?? null;
    this.path = opts.path ?? null;
    this.base64_data = opts.base64_data ?? null;
    this.media_type = opts.media_type ?? null;

    const sources = [this.url, this.path, this.base64_data].filter((value) => value !== null).length;
    if (sources !== 1) {
      throw new Error('Exactly one of url, path, or base64_data must be provided.');
    }
    if (this.base64_data !== null && this.media_type === null) {
      throw new Error('media_type is required when using base64_data.');
    }
  }

  to_openai_content_part(): OpenAIImageContentPart {
    if (this.url !== null) {
      return { type: 'image_url', image_url: { url: this.url } };
    }

    if (this.path !== null) {
      const media_type = this.media_type ?? _guess_media_type(this.path);
      const data = readFileSync(this.path).toString('base64');
      return { type: 'image_url', image_url: { url: `data:${media_type};base64,${data}` } };
    }

    if (this.base64_data === null || this.media_type === null) {
      throw new Error('Invalid image state.');
    }
    return { type: 'image_url', image_url: { url: `data:${this.media_type};base64,${this.base64_data}` } };
  }

  toJSON(): { __gepa_image: OpenAIImageContentPart } {
    return { __gepa_image: this.to_openai_content_part() };
  }
}
