import { Metadata } from "next";
import FarmerDashboard from "@/components/FarmerDashboard";

export const metadata: Metadata = {
  title: "Farmer Intelligence Dashboard | Crop Health AI",
  description:
    "Unified agricultural intelligence dashboard featuring crop condition, vegetation health indicators, potential stress signals, growth forecasts, spatial risk maps, and evidence-linked recommendations.",
};

export default function FarmerDashboardPage() {
  return <FarmerDashboard />;
}
