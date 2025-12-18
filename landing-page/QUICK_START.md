# Quick Start Guide

## ✅ What's Already Done

- ✅ Next.js project structure with TypeScript and Tailwind CSS
- ✅ All components built and styled
- ✅ Images moved to `/public` folder
- ✅ Netlify configuration ready
- ✅ SEO metadata configured
- ✅ Responsive design implemented
- ✅ All sections from your specifications included

## 🚀 Next Steps

### 1. Update Configuration (5 minutes)

Edit `app/constants.ts`:
```typescript
export const calendlyLink = 'https://calendly.com/your-actual-link'
export const contactEmail = 'your-email@easywif.com'
export const linkedInUrl = 'https://linkedin.com/company/your-company'
```

### 2. Install Dependencies

```bash
cd landing-page
npm install
```

### 3. Test Locally (Optional)

```bash
npm run dev
```

Visit http://localhost:3000

### 4. Build & Deploy

```bash
npm run build
```

Then deploy the `/out` folder to Netlify (see DEPLOYMENT.md for details)

## 📝 Customization Checklist

- [ ] Update `app/constants.ts` with your links
- [ ] Review and customize copy in component files
- [ ] Add your video embed in `components/VideoSection.tsx`
- [ ] Test the demo form submission
- [ ] Verify all images display correctly
- [ ] Test on mobile devices
- [ ] Connect your custom domain in Netlify

## 📁 Project Structure

```
landing-page/
├── app/
│   ├── constants.ts      ← Update your links here
│   ├── layout.tsx        ← SEO metadata
│   ├── page.tsx          ← Main page
│   └── globals.css       ← Global styles
├── components/            ← All section components
├── public/               ← Your images (already here!)
├── netlify.toml          ← Netlify config
└── package.json          ← Dependencies
```

## 🎨 Styling

- Colors: Edit `tailwind.config.js`
- Fonts: Already using Inter (Google Font)
- Spacing: All components use Tailwind utility classes

## 📞 Need Help?

- See `DEPLOYMENT.md` for detailed deployment instructions
- See `README.md` for full documentation

