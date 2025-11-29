# Nginx Configuration Fix for AI Analysis Report URLs

## What to Add

Add this location block to `/etc/nginx/nginx.conf` in the `server { server_name deepgis.org www.deepgis.org; }` section.

## Location in File

**Insert after line 233** (after the `/webclient/` block) and **before line 235** (before the default `location /` block).

## Code to Add

```nginx
        location /ai-analysis/ {
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header Host $host;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_pass http://localhost:8060;
            proxy_http_version 1.1;
        }
```

## Complete Context (Lines 227-241)

```nginx
        location /webclient/ {
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header Host $host;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_pass http://localhost:8060;
            proxy_http_version 1.1;
        }

        # ADD THIS BLOCK HERE (new location for AI analysis reports)
        location /ai-analysis/ {
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header Host $host;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_pass http://localhost:8060;
            proxy_http_version 1.1;
        }

        location / {
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header Host $host;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_pass http://localhost:8080;
            proxy_http_version 1.1;
        }
```

## Why This Order Matters

Nginx matches location blocks in order of specificity. More specific paths must come before less specific ones:

1. `/map-label/` → port 8060 (deepgis-xr)
2. `/label/` → port 8060 (deepgis-xr)
3. `/webclient/` → port 8060 (deepgis-xr)
4. **`/ai-analysis/` → port 8060 (deepgis-xr)** ← **ADD THIS**
5. `/` → port 8080 (dreams_laboratory) ← Default catch-all

## After Adding

1. Test nginx configuration:
   ```bash
   sudo nginx -t
   ```

2. If test passes, reload nginx:
   ```bash
   sudo systemctl reload nginx
   # OR
   sudo nginx -s reload
   ```

3. Test the URL:
   ```
   http://deepgis.org/ai-analysis/report/sam_20251128_213833_lat64p495971_lonn165p427112_alt251m_modelvit_b/
   ```

## What This Fixes

- **Before:** `/ai-analysis/` requests fall through to default `location /` → port 8080 → dreams_laboratory → 404 (no URL pattern)
- **After:** `/ai-analysis/` requests match the new location → port 8060 → deepgis-xr → ✅ Works (URL pattern exists)

