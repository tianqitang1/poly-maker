import { useState, useEffect } from 'react'
import './App.css'
import { StatsTable } from './components/StatsTable'
import { ConfigEditor } from './components/ConfigEditor'
import { MarketBrowser } from './components/MarketBrowser'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [stats, setStats] = useState([])
  const [allMarkets, setAllMarkets] = useState([])
  const [volMarkets, setVolMarkets] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAllData()
  }, [])

  const fetchAllData = async () => {
    setLoading(true)
    try {
      const [statsRes, allRes, volRes] = await Promise.all([
        fetch('http://localhost:8001/data/stats').catch(() => ({ json: () => [] })),
        fetch('http://localhost:8001/data/all_markets').catch(() => ({ json: () => [] })),
        fetch('http://localhost:8001/data/volatility_markets').catch(() => ({ json: () => [] }))
      ])

      const sData = await statsRes.json()
      const aData = await allRes.json()
      const vData = await volRes.json()

      setStats(sData)
      setAllMarkets(aData)
      setVolMarkets(vData)
    } catch (e) {
      console.error("Error fetching data", e)
    } finally {
      setLoading(false)
    }
  }
  
  const handleAddMarket = async (market) => {
      // This needs to add to the Selected Markets sheet via the backend
      // First fetch current selected
      try {
          const res = await fetch('http://localhost:8001/config/sheets/selected_markets');
          const current = await res.json();
          
          // Check if exists
          if (current.find(m => m.question === market.question)) {
              alert("Already in selected markets!");
              return;
          }
          
          const newEntry = {
              question: market.question,
              max_size: 200,
              trade_size: 50,
              param_type: "illiquid",
              multiplier: "",
              quote_offset_ticks: 2,
              comments: "Added via Dashboard"
          };
          
          // Save back
          const saveRes = await fetch('http://localhost:8001/config/sheets/selected_markets', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify([...current, newEntry])
          });
          
          if (saveRes.ok) alert("Market added to Sheets!");
      } catch (e) {
          alert("Error adding market: " + e);
      }
  }

  return (
    <div className="container">
      <div className="flex-between">
          <h1>Poly-Maker Dashboard</h1>
          <button className="btn" onClick={fetchAllData}>Refresh Data</button>
      </div>

      <div className="tabs">
        <button onClick={() => setActiveTab('dashboard')} className={activeTab === 'dashboard' ? 'active' : ''}>Dashboard</button>
        <button onClick={() => setActiveTab('markets')} className={activeTab === 'markets' ? 'active' : ''}>All Markets</button>
        <button onClick={() => setActiveTab('volatility')} className={activeTab === 'volatility' ? 'active' : ''}>Volatility</button>
        <button onClick={() => setActiveTab('config')} className={activeTab === 'config' ? 'active' : ''}>Configuration</button>
        <button onClick={() => setActiveTab('browse')} className={activeTab === 'browse' ? 'active' : ''}>Browse</button>
      </div>

      {activeTab === 'dashboard' && (
        <div>
           <div className="grid-2" style={{marginBottom: '20px'}}>
               <div className="stat-card">
                   <div className="stat-value">{stats.length}</div>
                   <div className="stat-label">Active Positions</div>
               </div>
               <div className="stat-card">
                   <div className="stat-value">
                       ${stats.reduce((acc, curr) => acc + (curr.earnings || 0), 0).toFixed(2)}
                   </div>
                   <div className="stat-label">Total Earnings (Est)</div>
               </div>
           </div>
           
           <StatsTable 
             title="Current Positions & Stats" 
             data={stats} 
             columns={[
                 { key: 'question', label: 'Market' },
                 { key: 'position_size', label: 'Position Size' },
                 { key: 'order_size', label: 'Order Size' },
                 { key: 'earnings', label: 'Earnings' },
                 { key: 'earning_percentage', label: 'Yield %' }
             ]}
           />
        </div>
      )}

      {activeTab === 'markets' && (
        <StatsTable 
            title="All Analyzed Markets" 
            data={allMarkets}
            columns={[
                'question', 'gm_reward_per_100', 'volatility_sum', 'spread', 'min_size'
            ]} 
        />
      )}

      {activeTab === 'volatility' && (
        <StatsTable 
            title="Volatility Markets" 
            data={volMarkets}
            columns={[
                'question', 'gm_reward_per_100', 'volatility_sum', 'best_bid', 'best_ask', 'volatility_price'
            ]}
        />
      )}

      {activeTab === 'config' && (
        <ConfigEditor />
      )}
      
      {activeTab === 'browse' && (
          <MarketBrowser onAddMarket={handleAddMarket} />
      )}
    </div>
  )
}

export default App